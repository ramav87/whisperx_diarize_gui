from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

# WhisperX and Pyannote 3.x load trusted Lightning checkpoints during normal
# ASR/diarization startup. PyTorch 2.6+ defaults torch.load to weights_only=True,
# which rejects those checkpoints unless this compatibility flag is set before
# either library imports.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from .pipeline import DEFAULT_OLLAMA_ANALYSIS_MODEL, DiarizationPipelineRunner
from .metrics.context_adjusted import build_context_metrics, interpretation_for_context_metrics
from .dashboard_metrics import build_profile_dashboard


logging.basicConfig(
    level=os.environ.get("DIARIZE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("diarize_server")


def _server_data_dir() -> Path:
    configured = os.environ.get("DIARIZE_SERVER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("~/.local/share/diarize-server").expanduser().resolve()


DATA_DIR = _server_data_dir()
UPLOADS_DIR = DATA_DIR / "uploads"
LESSONS_DIR = DATA_DIR / "lessons"
PROFILES_DIR = DATA_DIR / "profiles"

API_VERSION = "2026-06-28"
ARTIFACT_FILES = {
    "transcript": "transcript.txt",
    "transcript_raw": "transcript_raw.txt",
    "transcript_cleaned": "transcript_cleaned.txt",
    "transcript_highlighted": "transcript_cleaned_highlighted.txt",
    "segments": "segments.json",
    "segments_raw": "segments_raw.json",
    "diarization": "diarization.json",
    "transcript_artifact": "transcript_artifact.json",
    "ai_stats": "ai_stats.json",
    "analysis": "analysis.txt",
    "audio": "audio.wav",
    "normalized_audio": "normalized_audio.wav",
}


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
_analysis_jobs: dict[str, JobState] = {}
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
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        _load_jobs_from_disk()
        logger.info(
            "server startup data_dir=%s uploads_dir=%s lessons_dir=%s",
            DATA_DIR,
            UPLOADS_DIR,
            LESSONS_DIR,
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "data_dir": str(DATA_DIR),
            "uploads_dir": str(UPLOADS_DIR),
            "lessons_dir": str(LESSONS_DIR),
            "profiles_dir": str(PROFILES_DIR),
        }

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "job_statuses": ["queued", "running", "succeeded", "failed"],
            "artifact_names": sorted(ARTIFACT_FILES),
            "endpoints": {
                "health": "/api/health",
                "capabilities": "/api/capabilities",
                "create_processing_job": "/api/jobs",
                "list_processing_jobs": "/api/jobs",
                "get_processing_job": "/api/jobs/{job_id}",
                "list_profiles": "/api/profiles",
                "create_profile": "/api/profiles",
                "get_profile": "/api/profiles/{profile_id}",
                "update_profile": "/api/profiles/{profile_id}",
                "patch_profile": "/api/profiles/{profile_id}",
                "profile_dashboard": "/api/profiles/{profile_id}/dashboard",
                "profile_lessons": "/api/profiles/{profile_id}/lessons",
                "create_profile_processing_job": "/api/profiles/{profile_id}/jobs",
                "list_lessons": "/api/lessons",
                "get_lesson": "/api/lessons/{lesson_id}",
                "lesson_artifacts": "/api/lessons/{lesson_id}/artifacts",
                "lesson_artifact": "/api/lessons/{lesson_id}/artifacts/{artifact_name}",
                "lesson_context": "/api/lessons/{lesson_id}/context",
                "transcript_export": "/api/lessons/{lesson_id}/exports/transcript",
                "analysis_export": "/api/lessons/{lesson_id}/exports/analysis",
                "create_analysis_job": "/api/lessons/{lesson_id}/analysis-jobs",
                "get_analysis_job": "/api/analysis-jobs/{job_id}",
            },
        }

    @app.get("/api/profiles")
    def list_profiles() -> dict[str, Any]:
        profile_ids = _discover_profile_ids()
        return {"profiles": [_profile_response(profile_id) for profile_id in profile_ids]}

    @app.post("/api/profiles")
    def create_profile(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Profile payload must be an object")
        requested_id = payload.get("id") or payload.get("display_name")
        safe_profile = _safe_profile_id(str(requested_id or "default"))
        if _profile_path(safe_profile).exists():
            raise HTTPException(status_code=409, detail="Profile already exists")
        return _write_profile_payload(safe_profile, payload, merge_settings=False)

    @app.get("/api/profiles/{profile_id}")
    def get_profile(profile_id: str) -> dict[str, Any]:
        return _profile_response(_safe_profile_id(profile_id))

    @app.put("/api/profiles/{profile_id}")
    def update_profile(
        profile_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        return _write_profile_payload(_safe_profile_id(profile_id), payload, merge_settings=False)

    @app.patch("/api/profiles/{profile_id}")
    def patch_profile(
        profile_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        return _write_profile_payload(_safe_profile_id(profile_id), payload, merge_settings=True)

    @app.get("/api/profiles/{profile_id}/dashboard")
    def get_profile_dashboard(profile_id: str) -> dict[str, Any]:
        return build_profile_dashboard(_safe_profile_id(profile_id), LESSONS_DIR)

    @app.get("/api/profiles/{profile_id}/lessons")
    def list_profile_lessons(profile_id: str) -> dict[str, Any]:
        safe_profile = _safe_profile_id(profile_id)
        _ensure_profile(safe_profile)
        return {"profile": safe_profile, "lessons": _lesson_summaries(safe_profile)}

    @app.post("/api/profiles/{profile_id}/jobs")
    async def create_profile_processing_job(
        profile_id: str,
        audio: UploadFile = File(...),
        model_size: str = Form("large-v3"),
        language: Optional[str] = Form(None),
        num_speakers: Optional[int] = Form(None),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
        backend: str = Form("auto"),
        diarization_backend: str = Form("auto"),
        batch_size: Optional[int] = Form(None),
    ) -> dict[str, Any]:
        return _create_processing_job(
            audio=audio,
            profile=_safe_profile_id(profile_id),
            model_size=model_size,
            language=language,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            backend=backend,
            diarization_backend=diarization_backend,
            batch_size=batch_size,
        )

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
        diarization_backend: str = Form("auto"),
        batch_size: Optional[int] = Form(None),
    ) -> dict[str, Any]:
        return _create_processing_job(
            audio=audio,
            profile=profile,
            model_size=model_size,
            language=language,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            backend=backend,
            diarization_backend=diarization_backend,
            batch_size=batch_size,
        )

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        with _jobs_lock:
            jobs = sorted(_jobs.values(), key=lambda item: item.created_at, reverse=True)
        return {"jobs": [_job_response(job) for job in jobs]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _job_response(_get_job_or_404(job_id))

    @app.get("/api/lessons")
    def list_lessons(profile: Optional[str] = Query(None)) -> dict[str, Any]:
        safe_profile = _safe_profile_id(profile) if profile else None
        return {"lessons": _lesson_summaries(safe_profile)}

    @app.get("/api/lessons/{lesson_id}")
    def get_lesson(lesson_id: str) -> dict[str, Any]:
        return _lesson_response(lesson_id, _lesson_dir_or_404(lesson_id))

    @app.get("/api/lessons/{lesson_id}/artifacts")
    def list_lesson_artifacts(lesson_id: str) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        return {"lesson_id": lesson_id, "artifacts": _lesson_artifact_summary(lesson_dir)}

    @app.get("/api/lessons/{lesson_id}/artifacts/{artifact_name}")
    def get_lesson_artifact(lesson_id: str, artifact_name: str):
        lesson_dir = _lesson_dir_or_404(lesson_id)
        artifact_path = _artifact_path_or_404(lesson_dir, artifact_name)
        media_type = _artifact_media_type(artifact_path)
        if media_type.startswith("text/"):
            return PlainTextResponse(_read_text(artifact_path) or "", media_type=media_type)
        return FileResponse(artifact_path, media_type=media_type, filename=artifact_path.name)

    @app.get("/api/lessons/{lesson_id}/exports/transcript")
    def export_lesson_transcript(
        lesson_id: str,
        variant: str = Query("cleaned"),
        format: str = Query("txt"),
    ):
        if variant not in {"cleaned", "raw", "highlighted"}:
            raise HTTPException(status_code=400, detail="variant must be cleaned, raw, or highlighted")
        if format not in {"txt", "json"}:
            raise HTTPException(status_code=400, detail="format must be txt or json")
        lesson_dir = _lesson_dir_or_404(lesson_id)
        artifact_name = {
            ("cleaned", "txt"): "transcript_cleaned",
            ("raw", "txt"): "transcript_raw",
            ("highlighted", "txt"): "transcript_highlighted",
            ("cleaned", "json"): "segments",
            ("raw", "json"): "segments_raw",
            ("highlighted", "json"): "segments",
        }[(variant, format)]
        artifact_path = _artifact_path_or_404(lesson_dir, artifact_name)
        media_type = "application/json" if format == "json" else "text/plain; charset=utf-8"
        filename = f"{lesson_id}_transcript_{variant}.{format}"
        return FileResponse(artifact_path, media_type=media_type, filename=filename)

    @app.get("/api/lessons/{lesson_id}/exports/analysis")
    def export_lesson_analysis(
        lesson_id: str,
        format: str = Query("txt"),
    ):
        if format not in {"txt", "json"}:
            raise HTTPException(status_code=400, detail="format must be txt or json")
        lesson_dir = _lesson_dir_or_404(lesson_id)
        if format == "json":
            artifact_path = _artifact_path_or_404(lesson_dir, "ai_stats")
            return FileResponse(
                artifact_path,
                media_type="application/json",
                filename=f"{lesson_id}_analysis.json",
            )
        artifact_path = _artifact_path_or_404(lesson_dir, "analysis")
        return FileResponse(
            artifact_path,
            media_type="text/plain; charset=utf-8",
            filename=f"{lesson_id}_analysis.txt",
        )

    @app.get("/api/lessons/{lesson_id}/context")
    def get_lesson_context(lesson_id: str) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        return _lesson_context_response(lesson_id, lesson_dir)

    @app.patch("/api/lessons/{lesson_id}/context")
    def update_lesson_context(
        lesson_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Context payload must be an object")
        ai_stats = _read_json(lesson_dir / "ai_stats.json") or {}
        if not isinstance(ai_stats, dict):
            ai_stats = {}
        current = ai_stats.get("context_metrics") if isinstance(ai_stats.get("context_metrics"), dict) else {}
        merged = dict(current)
        merged.update(payload)
        updated = _build_context_from_payload(ai_stats, merged, lesson_dir)
        ai_stats["context_metrics"] = updated
        for key in (
            "topic_difficulty",
            "idea_density",
            "abstraction_level",
            "cognitive_branching",
            "technical_density",
            "discourse_depth",
            "lexical_retrieval_pressure",
        ):
            if updated.get(key) is not None:
                ai_stats[key] = updated.get(key)
        ai_stats["context_updated_at"] = _now()
        _write_json(lesson_dir / "ai_stats.json", ai_stats)
        meta = _read_json(lesson_dir / "meta.json") or {}
        meta["context_updated_at"] = ai_stats["context_updated_at"]
        _write_json(lesson_dir / "meta.json", meta)
        return _lesson_context_response(lesson_id, lesson_dir)

    @app.patch("/api/lessons/{lesson_id}/speakers")
    def update_lesson_speakers(
        lesson_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        meta_path = lesson_dir / "meta.json"
        segments = _read_json(lesson_dir / "segments.json") or []
        known_speakers = {str(seg.get("speaker")) for seg in segments if seg.get("speaker")}

        speaker_labels = payload.get("speaker_labels", {})
        if speaker_labels is None:
            speaker_labels = {}
        if not isinstance(speaker_labels, dict):
            raise HTTPException(status_code=400, detail="speaker_labels must be an object")

        normalized_labels = {
            str(speaker): str(label).strip()
            for speaker, label in speaker_labels.items()
            if str(speaker).strip() and str(label).strip()
        }
        unknown_labels = sorted(set(normalized_labels) - known_speakers)
        if unknown_labels:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown speaker id(s): {', '.join(unknown_labels)}",
            )

        student_speakers = payload.get("student_speakers", [])
        if student_speakers is None:
            student_speakers = []
        if not isinstance(student_speakers, list):
            raise HTTPException(status_code=400, detail="student_speakers must be a list")

        normalized_students = [str(speaker) for speaker in student_speakers if str(speaker).strip()]
        unknown_students = sorted(set(normalized_students) - known_speakers)
        if unknown_students:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown student speaker id(s): {', '.join(unknown_students)}",
            )

        meta = _read_json(meta_path) or {}
        meta["speaker_labels"] = normalized_labels
        meta["student_speakers"] = normalized_students
        meta["speaker_reviewed_at"] = _now()
        _write_json(meta_path, meta)
        return _lesson_response(lesson_id, lesson_dir)

    @app.post("/api/lessons/{lesson_id}/analyze")
    def analyze_lesson(
        lesson_id: str,
        payload: Optional[dict[str, Any]] = Body(None),
    ) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        try:
            _compute_lesson_analysis(lesson_dir, payload or {})
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _lesson_response(lesson_id, lesson_dir)

    @app.post("/api/lessons/{lesson_id}/analysis-jobs")
    def create_analysis_job(
        lesson_id: str,
        payload: Optional[dict[str, Any]] = Body(None),
    ) -> dict[str, Any]:
        lesson_dir = _lesson_dir_or_404(lesson_id)
        job_id = uuid.uuid4().hex
        now = _now()
        state = JobState(
            id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            lesson_id=lesson_id,
            message="Queued analysis",
        )
        with _jobs_lock:
            _analysis_jobs[job_id] = state
        _persist_job("analysis", state)
        provider = str((payload or {}).get("provider") or os.environ.get("DIARIZE_LLM_PROVIDER") or "ollama")
        model = (payload or {}).get("model") or os.environ.get("DIARIZE_LLM_MODEL") or DEFAULT_OLLAMA_ANALYSIS_MODEL
        logger.info(
            "analysis job queued job_id=%s lesson_id=%s provider=%s model=%s",
            job_id,
            lesson_id,
            provider,
            model,
        )

        _executor.submit(
            _run_analysis_job,
            job_id=job_id,
            lesson_id=lesson_id,
            lesson_dir=lesson_dir,
            payload=payload or {},
        )
        return _job_response(state)

    @app.get("/api/analysis-jobs/{job_id}")
    def get_analysis_job(job_id: str) -> dict[str, Any]:
        return _job_response(_get_analysis_job_or_404(job_id))

    @app.get("/api/analysis-jobs")
    def list_analysis_jobs() -> dict[str, Any]:
        with _jobs_lock:
            jobs = sorted(_analysis_jobs.values(), key=lambda item: item.created_at, reverse=True)
        return {"jobs": [_job_response(job) for job in jobs]}

    return app


app = create_app()


def _create_processing_job(
    *,
    audio: UploadFile,
    profile: str,
    model_size: str,
    language: Optional[str],
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    backend: str,
    diarization_backend: str,
    batch_size: Optional[int],
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    safe_name = _safe_upload_name(audio.filename or "audio")
    upload_path = UPLOADS_DIR / f"{job_id}_{safe_name}"

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as f:
        shutil.copyfileobj(audio.file, f)
    try:
        upload_size = upload_path.stat().st_size
    except OSError:
        upload_size = None
    safe_profile = _ensure_profile(profile)
    logger.info(
        "processing job queued job_id=%s profile=%s filename=%s bytes=%s model=%s language=%s backend=%s diarization=%s",
        job_id,
        safe_profile,
        audio.filename,
        upload_size,
        model_size,
        language,
        backend,
        diarization_backend,
    )

    now = _now()
    state = JobState(id=job_id, status="queued", created_at=now, updated_at=now)
    with _jobs_lock:
        _jobs[job_id] = state
    _persist_job("processing", state)

    _executor.submit(
        _run_processing_job,
        job_id=job_id,
        upload_path=upload_path,
        profile=safe_profile,
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
    started = time.monotonic()
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
    logger.info(
        "processing job started job_id=%s lesson_id=%s upload_path=%s output_dir=%s",
        job_id,
        lesson_id,
        upload_path,
        lesson_dir,
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
        logger.info(
            "processing job succeeded job_id=%s lesson_id=%s elapsed_s=%.1f segments=%s speakers=%s",
            job_id,
            lesson_id,
            time.monotonic() - started,
            meta.get("num_segments"),
            meta.get("num_speakers"),
        )
    except Exception as exc:
        logger.exception("processing job failed job_id=%s lesson_id=%s", job_id, lesson_id)
        _update_job(
            job_id,
            status="failed",
            message="Processing failed",
            error=str(exc),
        )


def _run_analysis_job(
    *,
    job_id: str,
    lesson_id: str,
    lesson_dir: Path,
    payload: dict[str, Any],
) -> None:
    started = time.monotonic()
    _update_analysis_job(
        job_id,
        status="running",
        progress=1,
        message="Starting AI analysis",
    )
    provider = str(payload.get("provider") or os.environ.get("DIARIZE_LLM_PROVIDER") or "ollama")
    model = payload.get("model") or os.environ.get("DIARIZE_LLM_MODEL") or DEFAULT_OLLAMA_ANALYSIS_MODEL
    logger.info(
        "analysis job started job_id=%s lesson_id=%s provider=%s model=%s",
        job_id,
        lesson_id,
        provider,
        model,
    )

    try:
        result = _compute_lesson_analysis(
            lesson_dir,
            payload,
            status_callback=lambda message: _update_analysis_job(job_id, message=message),
            progress_callback=lambda progress: _update_analysis_job(
                job_id,
                progress=max(1.0, min(99.0, float(progress))),
            ),
        )
        _update_analysis_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Done",
            result={
                "lesson_id": lesson_id,
                "ai_stats": result.get("ai_stats"),
                "meta": result.get("meta"),
            },
        )
        logger.info(
            "analysis job succeeded job_id=%s lesson_id=%s elapsed_s=%.1f provider=%s model=%s",
            job_id,
            lesson_id,
            time.monotonic() - started,
            provider,
            model,
        )
    except Exception as exc:
        logger.exception("analysis job failed job_id=%s lesson_id=%s", job_id, lesson_id)
        _update_analysis_job(
            job_id,
            status="failed",
            message="AI analysis failed",
            error=str(exc),
        )


def _update_job(job_id: str, **updates: Any) -> None:
    state = _update_job_map(_jobs, job_id, **updates)
    if state:
        _persist_job("processing", state)


def _update_analysis_job(job_id: str, **updates: Any) -> None:
    state = _update_job_map(_analysis_jobs, job_id, **updates)
    if state:
        _persist_job("analysis", state)


def _update_job_map(job_map: dict[str, JobState], job_id: str, **updates: Any) -> Optional[JobState]:
    with _jobs_lock:
        state = job_map.get(job_id)
        if not state:
            return None
        old_status = state.status
        old_message = state.message
        old_progress = state.progress
        for key, value in updates.items():
            setattr(state, key, value)
        state.updated_at = _now()
        if (
            state.status != old_status
            or state.message != old_message
            or int(state.progress) != int(old_progress)
        ):
            logger.info(
                "job update job_id=%s status=%s progress=%.1f lesson_id=%s message=%s",
                job_id,
                state.status,
                state.progress,
                state.lesson_id,
                state.message,
            )
        return state


def _get_job_or_404(job_id: str) -> JobState:
    with _jobs_lock:
        state = _jobs.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


def _get_analysis_job_or_404(job_id: str) -> JobState:
    with _jobs_lock:
        state = _analysis_jobs.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return state


def _job_response(state: JobState) -> dict[str, Any]:
    return asdict(state)


def _jobs_dir(kind: str) -> Path:
    return DATA_DIR / "jobs" / kind


def _safe_profile_id(profile: Optional[str]) -> str:
    raw = str(profile or "default").strip() or "default"
    keep = []
    for char in raw:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    safe = "".join(keep).strip("_-")
    if not safe:
        safe = "default"
    if safe in {".", ".."}:
        safe = "default"
    return safe[:80]


def _profile_path(profile_id: str) -> Path:
    safe_profile = _safe_profile_id(profile_id)
    return PROFILES_DIR / f"{safe_profile}.json"


def _ensure_profile(profile_id: Optional[str], *, settings: Optional[dict[str, Any]] = None) -> str:
    safe_profile = _safe_profile_id(profile_id)
    path = _profile_path(safe_profile)
    existing = _read_json(path) or {}
    if not isinstance(existing, dict):
        existing = {}
    now = _now()
    merged_settings = existing.get("settings") if isinstance(existing.get("settings"), dict) else {}
    if settings:
        merged_settings = {**merged_settings, **settings}
    profile = {
        "id": safe_profile,
        "display_name": existing.get("display_name") or safe_profile,
        "settings": merged_settings,
        "created_at": existing.get("created_at") or now,
        "updated_at": now if settings else existing.get("updated_at") or now,
    }
    _write_json(path, profile)
    return safe_profile


def _write_profile_payload(
    profile_id: str,
    payload: dict[str, Any],
    *,
    merge_settings: bool,
) -> dict[str, Any]:
    safe_profile = _safe_profile_id(profile_id)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Profile payload must be an object")

    existing = _read_json(_profile_path(safe_profile)) or {}
    if not isinstance(existing, dict):
        existing = {}

    incoming_settings = payload.get("settings")
    if incoming_settings is None:
        settings = existing.get("settings", {}) if merge_settings else {}
    elif not isinstance(incoming_settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    elif merge_settings:
        current = existing.get("settings") if isinstance(existing.get("settings"), dict) else {}
        settings = {**current, **incoming_settings}
    else:
        settings = incoming_settings

    owner_id = payload.get("owner_id", existing.get("owner_id"))
    profile = {
        "id": safe_profile,
        "display_name": str(payload.get("display_name") or existing.get("display_name") or safe_profile),
        "settings": settings,
        "owner_id": owner_id,
        "created_at": existing.get("created_at") or _now(),
        "updated_at": _now(),
    }
    _write_json(_profile_path(safe_profile), profile)
    return _profile_response(safe_profile)


def _discover_profile_ids() -> list[str]:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_ids = set()
    for path in PROFILES_DIR.glob("*.json"):
        profile_ids.add(path.stem)
    if LESSONS_DIR.is_dir():
        for lesson_dir in LESSONS_DIR.iterdir():
            if not lesson_dir.is_dir():
                continue
            meta = _read_json(lesson_dir / "meta.json") or {}
            if isinstance(meta, dict) and meta.get("profile"):
                profile_ids.add(_safe_profile_id(meta.get("profile")))
    return sorted(profile_ids)


def _profile_response(profile_id: str) -> dict[str, Any]:
    safe_profile = _safe_profile_id(profile_id)
    data = _read_json(_profile_path(safe_profile)) or {}
    if not isinstance(data, dict):
        data = {}
    lesson_count = 0
    last_lesson_at = None
    if LESSONS_DIR.is_dir():
        for lesson_dir in LESSONS_DIR.iterdir():
            if not lesson_dir.is_dir():
                continue
            meta = _read_json(lesson_dir / "meta.json") or {}
            if not isinstance(meta, dict) or meta.get("profile") != safe_profile:
                continue
            lesson_count += 1
            candidate = meta.get("processed_at") or meta.get("recorded_at") or lesson_dir.name
            if candidate and (last_lesson_at is None or str(candidate) > str(last_lesson_at)):
                last_lesson_at = candidate
    return {
        "id": safe_profile,
        "display_name": data.get("display_name") or safe_profile,
        "settings": data.get("settings") if isinstance(data.get("settings"), dict) else {},
        "owner_id": data.get("owner_id"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "lesson_count": lesson_count,
        "last_lesson_at": last_lesson_at,
    }


def _lesson_summaries(profile: Optional[str] = None) -> list[dict[str, Any]]:
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    lessons = []
    for lesson_dir in sorted(LESSONS_DIR.iterdir(), reverse=True):
        if not lesson_dir.is_dir():
            continue
        meta = _read_json(lesson_dir / "meta.json") or {}
        if not isinstance(meta, dict):
            meta = {}
        if profile and meta.get("profile") != profile:
            continue
        lessons.append(
            {
                "id": lesson_dir.name,
                "profile": meta.get("profile"),
                "processed_at": meta.get("processed_at"),
                "recorded_at": meta.get("recorded_at"),
                "duration_sec": meta.get("duration_sec"),
                "num_segments": meta.get("num_segments"),
                "num_speakers": meta.get("num_speakers"),
                "artifacts": _lesson_artifact_summary(lesson_dir),
            }
        )
    return lessons


def _persist_job(kind: str, state: JobState) -> None:
    _write_json(_jobs_dir(kind) / f"{state.id}.json", _job_response(state))


def _load_jobs_from_disk() -> None:
    _load_job_dir("processing", _jobs)
    _load_job_dir("analysis", _analysis_jobs)


def _load_job_dir(kind: str, target: dict[str, JobState]) -> None:
    jobs_dir = _jobs_dir(kind)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    with _jobs_lock:
        target.clear()
        for path in sorted(jobs_dir.glob("*.json")):
            data = _read_json(path)
            if not isinstance(data, dict):
                continue
            try:
                state = JobState(
                    id=str(data["id"]),
                    status=str(data.get("status") or "failed"),
                    created_at=str(data.get("created_at") or _now()),
                    updated_at=str(data.get("updated_at") or _now()),
                    progress=float(data.get("progress") or 0),
                    message=str(data.get("message") or ""),
                    lesson_id=data.get("lesson_id"),
                    error=data.get("error"),
                    result=data.get("result") if isinstance(data.get("result"), dict) else {},
                )
            except (KeyError, TypeError, ValueError):
                continue
            if state.status in {"queued", "running"}:
                state.status = "failed"
                state.updated_at = _now()
                state.error = state.error or "Server restarted before this job completed."
                state.message = "Interrupted by server restart"
                _write_json(path, _job_response(state))
            target[state.id] = state


def _lesson_dir_or_404(lesson_id: str) -> Path:
    if "/" in lesson_id or "\\" in lesson_id or lesson_id in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="Lesson not found")
    lesson_dir = (LESSONS_DIR / lesson_id).resolve()
    if not str(lesson_dir).startswith(str(LESSONS_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Lesson not found")
    if not lesson_dir.is_dir():
        raise HTTPException(status_code=404, detail="Lesson not found")
    return lesson_dir


def _artifact_path_or_404(lesson_dir: Path, artifact_name: str) -> Path:
    filename = ARTIFACT_FILES.get(artifact_name)
    if not filename:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = (lesson_dir / filename).resolve()
    if not str(path).startswith(str(lesson_dir.resolve())):
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return path


def _artifact_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".txt", ".srt"}:
        return "text/plain; charset=utf-8"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def _lesson_artifact_summary(lesson_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for name, filename in sorted(ARTIFACT_FILES.items()):
        path = lesson_dir / filename
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        artifacts.append(
            {
                "name": name,
                "filename": filename,
                "media_type": _artifact_media_type(path),
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return artifacts


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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _lesson_response(lesson_id: str, lesson_dir: Path) -> dict[str, Any]:
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
        "artifacts": _lesson_artifact_summary(lesson_dir),
    }


def _lesson_context_response(lesson_id: str, lesson_dir: Path) -> dict[str, Any]:
    ai_stats = _read_json(lesson_dir / "ai_stats.json") or {}
    if not isinstance(ai_stats, dict):
        ai_stats = {}
    context = ai_stats.get("context_metrics") if isinstance(ai_stats.get("context_metrics"), dict) else {}
    if not context:
        context = _build_context_from_payload(ai_stats, {}, lesson_dir)
    return {
        "lesson_id": lesson_id,
        "context_metrics": context,
        "interpretation": interpretation_for_context_metrics(context),
        "ai_stats": ai_stats or None,
    }


def _build_context_from_payload(
    ai_stats: dict[str, Any],
    payload: dict[str, Any],
    lesson_dir: Path,
) -> dict[str, Any]:
    raw_wpm = payload.get("raw_wpm")
    if raw_wpm is None:
        raw_wpm = _compute_lesson_raw_wpm(lesson_dir)
    return build_context_metrics(
        raw_grammar_score=ai_stats.get("grammar_score", payload.get("raw_grammar_score")),
        raw_wpm=raw_wpm,
        practice_hours_last_7_days=payload.get("practice_hours_last_7_days"),
        topic_difficulty=payload.get("topic_difficulty", ai_stats.get("topic_difficulty")),
        idea_density=payload.get("idea_density", ai_stats.get("idea_density")),
        abstraction_level=payload.get("abstraction_level", ai_stats.get("abstraction_level")),
        cognitive_branching=payload.get("cognitive_branching", ai_stats.get("cognitive_branching")),
        technical_density=payload.get("technical_density", ai_stats.get("technical_density")),
        discourse_depth=payload.get("discourse_depth", ai_stats.get("discourse_depth")),
        lexical_retrieval_pressure=payload.get(
            "lexical_retrieval_pressure",
            ai_stats.get("lexical_retrieval_pressure"),
        ),
        fatigue_or_stress=payload.get("fatigue_or_stress"),
        long_pauses_per_min=payload.get("long_pauses_per_min"),
        self_repairs_per_min=payload.get("self_repairs_per_min"),
        filled_pauses_per_min=payload.get("filled_pauses_per_min"),
        notes=payload.get("notes", ai_stats.get("context_notes")),
    )


def _compute_lesson_raw_wpm(lesson_dir: Path) -> Optional[float]:
    segments = _read_json(lesson_dir / "segments.json") or []
    meta = _read_json(lesson_dir / "meta.json") or {}
    if not isinstance(segments, list) or not isinstance(meta, dict):
        return None
    student_ids = {str(speaker) for speaker in meta.get("student_speakers", [])}
    words = 0
    seconds = 0.0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        speaker = str(seg.get("speaker", "UNKNOWN"))
        is_student = (speaker in student_ids) or (not student_ids and "01" in speaker)
        if not is_student:
            continue
        try:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
        except (TypeError, ValueError):
            continue
        seconds += max(0.0, end - start)
        words += len(str(seg.get("text", "")).strip().split())
    return (words / (seconds / 60.0)) if seconds > 10 else None


def _compute_lesson_analysis(
    lesson_dir: Path,
    payload: dict[str, Any],
    *,
    status_callback=None,
    progress_callback=None,
) -> dict[str, Any]:
    provider = str(payload.get("provider") or os.environ.get("DIARIZE_LLM_PROVIDER") or "ollama")
    provider = provider.strip().lower()
    model = payload.get("model") or os.environ.get("DIARIZE_LLM_MODEL") or DEFAULT_OLLAMA_ANALYSIS_MODEL
    api_key = payload.get("api_key") or os.environ.get("OPENAI_API_KEY")
    api_url = payload.get("api_url") or _default_ollama_api_url()
    logger.info(
        "analysis compute started lesson_id=%s provider=%s model=%s api_url=%s",
        lesson_dir.name,
        provider,
        model,
        api_url if provider == "ollama" else None,
    )

    runner = DiarizationPipelineRunner(
        status_callback=status_callback,
        progress_callback=progress_callback,
    )
    runner.load_lesson_artifacts(str(lesson_dir))
    success = runner.compute_ai_metrics(
        str(lesson_dir),
        model=model or None,
        mode=provider,
        api_key=api_key,
        api_url=api_url if provider == "ollama" else None,
    )
    if not success:
        raise RuntimeError(runner.last_ai_metrics_error or "AI analysis failed")

    meta_path = lesson_dir / "meta.json"
    meta = _read_json(meta_path) or {}
    meta["llm_provider"] = provider
    meta["llm_model"] = model
    meta["analyzed_at"] = _now()
    _write_json(meta_path, meta)
    logger.info("analysis compute saved lesson_id=%s ai_stats=%s", lesson_dir.name, lesson_dir / "ai_stats.json")
    return _lesson_response(lesson_dir.name, lesson_dir)


def _default_ollama_api_url() -> str:
    configured = os.environ.get("DIARIZE_OLLAMA_API_URL")
    if configured:
        return configured

    host = os.environ.get("OLLAMA_HOST")
    if host:
        base = host if "://" in host else f"http://{host}"
        return base.rstrip("/") + "/api/generate"

    return "http://127.0.0.1:11434/api/generate"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    import uvicorn

    host = os.environ.get("DIARIZE_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("DIARIZE_SERVER_PORT", "8000"))
    log_level = os.environ.get("DIARIZE_LOG_LEVEL", "info").lower()
    logger.info("starting uvicorn host=%s port=%s log_level=%s", host, port, log_level)
    uvicorn.run("diarize_gui.api:app", host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
