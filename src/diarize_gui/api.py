from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# WhisperX and Pyannote 3.x load trusted Lightning checkpoints during normal
# ASR/diarization startup. PyTorch 2.6+ defaults torch.load to weights_only=True,
# which rejects those checkpoints unless this compatibility flag is set before
# either library imports.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from .pipeline import DiarizationPipelineRunner


def _server_data_dir() -> Path:
    configured = os.environ.get("DIARIZE_SERVER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("~/.local/share/diarize-server").expanduser().resolve()


DATA_DIR = _server_data_dir()
UPLOADS_DIR = DATA_DIR / "uploads"
LESSONS_DIR = DATA_DIR / "lessons"


@dataclass
class JobState:
    id: str
    status: str
    created_at: str
    updated_at: str
    progress: float = 0.0
    message: str = "Queued"
    lesson_id: Optional[str] = None
    error: Optional[str] = None
    result: dict[str, Any] = field(default_factory=dict)


_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def create_app() -> FastAPI:
    app = FastAPI(title="Diarize Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _ensure_storage() -> None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        LESSONS_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "data_dir": str(DATA_DIR),
            "uploads_dir": str(UPLOADS_DIR),
            "lessons_dir": str(LESSONS_DIR),
        }

    @app.post("/api/jobs")
    async def create_job(
        audio: UploadFile = File(...),
        profile: str = Form("default"),
        model_size: str = Form("large-v3"),
        language: Optional[str] = Form(None),
        num_speakers: Optional[int] = Form(None),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
        backend: str = Form("auto"),
        diarization_backend: str = Form("pyannote"),
        batch_size: Optional[int] = Form(None),
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        safe_name = _safe_upload_name(audio.filename or "audio")
        upload_path = UPLOADS_DIR / f"{job_id}_{safe_name}"

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        with upload_path.open("wb") as f:
            shutil.copyfileobj(audio.file, f)

        now = _now()
        state = JobState(id=job_id, status="queued", created_at=now, updated_at=now)
        with _jobs_lock:
            _jobs[job_id] = state

        _executor.submit(
            _run_processing_job,
            job_id=job_id,
            upload_path=upload_path,
            profile=profile,
            model_size=model_size,
            language=language or None,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            backend=backend,
            diarization_backend=diarization_backend,
            batch_size=batch_size,
        )
        return _job_response(state)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _job_response(_get_job_or_404(job_id))

    @app.get("/api/lessons")
    def list_lessons() -> dict[str, Any]:
        LESSONS_DIR.mkdir(parents=True, exist_ok=True)
        lessons = []
        for lesson_dir in sorted(LESSONS_DIR.iterdir(), reverse=True):
            if not lesson_dir.is_dir():
                continue
            meta = _read_json(lesson_dir / "meta.json") or {}
            lessons.append(
                {
                    "id": lesson_dir.name,
                    "profile": meta.get("profile"),
                    "processed_at": meta.get("processed_at"),
                    "duration_sec": meta.get("duration_sec"),
                    "num_segments": meta.get("num_segments"),
                    "num_speakers": meta.get("num_speakers"),
                }
            )
        return {"lessons": lessons}

    @app.get("/api/lessons/{lesson_id}")
    def get_lesson(lesson_id: str) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        meta = _read_json(lesson_dir / "meta.json") or {}
        segments = _read_json(lesson_dir / "segments.json") or []
        ai_stats = _read_json(lesson_dir / "ai_stats.json")
        transcript = _read_text(lesson_dir / "transcript.txt")
        return {
            "id": lesson_id,
            "meta": meta,
            "segments": segments,
            "transcript": transcript,
            "ai_stats": ai_stats,
        }

    return app


app = create_app()


def _run_processing_job(
    *,
    job_id: str,
    upload_path: Path,
    profile: str,
    model_size: str,
    language: Optional[str],
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    backend: str,
    diarization_backend: str,
    batch_size: Optional[int],
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lesson_id = f"{timestamp}_{job_id[:8]}"
    lesson_dir = LESSONS_DIR / lesson_id

    def set_status(message: str) -> None:
        _update_job(job_id, message=message)

    def set_progress(progress: float) -> None:
        _update_job(job_id, progress=max(0.0, min(100.0, float(progress))))

    _update_job(
        job_id,
        status="running",
        message="Starting transcription and diarization",
        lesson_id=lesson_id,
        progress=1,
    )

    try:
        lesson_dir.mkdir(parents=True, exist_ok=True)
        runner = DiarizationPipelineRunner(
            status_callback=set_status,
            progress_callback=set_progress,
        )
        txt_path, json_path = runner.process_audio(
            str(upload_path),
            str(lesson_dir),
            model_size=model_size,
            language=language,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            backend=backend,
            diarization_backend=diarization_backend,
            batch_size=batch_size,
        )
        meta = runner.save_lesson_artifacts(
            str(lesson_dir),
            profile_name=profile,
            whisper_model_size=model_size,
            language=language,
            extra_meta={
                "server_job_id": job_id,
                "uploaded_filename": upload_path.name,
            },
        )
        _update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Done",
            lesson_id=lesson_id,
            result={
                "lesson_id": lesson_id,
                "transcript_path": txt_path,
                "result_path": json_path,
                "meta": meta,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            message="Processing failed",
            error=str(exc),
        )


def _update_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        state = _jobs.get(job_id)
        if not state:
            return
        for key, value in updates.items():
            setattr(state, key, value)
        state.updated_at = _now()


def _get_job_or_404(job_id: str) -> JobState:
    with _jobs_lock:
        state = _jobs.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


def _job_response(state: JobState) -> dict[str, Any]:
    return asdict(state)


def _lesson_dir_or_404(lesson_id: str) -> Path:
    if "/" in lesson_id or "\\" in lesson_id or lesson_id in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="Lesson not found")
    lesson_dir = (LESSONS_DIR / lesson_id).resolve()
    if not str(lesson_dir).startswith(str(LESSONS_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Lesson not found")
    if not lesson_dir.is_dir():
        raise HTTPException(status_code=404, detail="Lesson not found")
    return lesson_dir


def _safe_upload_name(filename: str) -> str:
    keep = []
    for char in Path(filename).name:
        if char.isalnum() or char in {".", "-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    safe = "".join(keep).strip("._")
    return safe or "audio"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    import uvicorn

    host = os.environ.get("DIARIZE_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("DIARIZE_SERVER_PORT", "8000"))
    uvicorn.run("diarize_gui.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
