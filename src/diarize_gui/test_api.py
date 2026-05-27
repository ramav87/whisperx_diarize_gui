import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import diarize_gui.api as api


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


class FakePipelineRunner:
    def __init__(self, status_callback=None, progress_callback=None):
        self.status_callback = status_callback
        self.progress_callback = progress_callback

    def process_audio(self, audio_path, output_dir, **kwargs):
        if self.status_callback:
            self.status_callback("Fake processing")
        if self.progress_callback:
            self.progress_callback(42)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        txt_path = out / "uploaded_diarized.txt"
        json_path = out / "uploaded_diarized.json"
        txt_path.write_text("SPEAKER_00: hola\n", encoding="utf-8")
        json_path.write_text(
            json.dumps({"segments": [{"speaker": "SPEAKER_00", "text": "hola"}]}),
            encoding="utf-8",
        )
        return str(txt_path), str(json_path)

    def save_lesson_artifacts(self, lesson_dir, **kwargs):
        out = Path(lesson_dir)
        meta = {
            "profile": kwargs.get("profile_name"),
            "processed_at": "2026-05-25T12:00:00",
            "duration_sec": 1.0,
            "num_segments": 1,
            "num_speakers": 1,
            "server_job_id": kwargs.get("extra_meta", {}).get("server_job_id"),
        }
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (out / "segments.json").write_text(
            json.dumps([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hola"}]),
            encoding="utf-8",
        )
        (out / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")
        return meta

    def load_lesson_artifacts(self, lesson_dir):
        self.lesson_dir = Path(lesson_dir)
        return json.loads((self.lesson_dir / "meta.json").read_text(encoding="utf-8"))

    def compute_ai_metrics(self, lesson_dir, **kwargs):
        out = Path(lesson_dir)
        (out / "ai_stats.json").write_text(
            json.dumps(
                {
                    "grammar_score": 88,
                    "llm_provider": kwargs.get("mode"),
                    "llm_model": kwargs.get("model"),
                }
            ),
            encoding="utf-8",
        )
        return True


class FailingPipelineRunner(FakePipelineRunner):
    def compute_ai_metrics(self, lesson_dir, **kwargs):
        self.last_ai_metrics_error = "Error: model not found"
        return False


class ApiTests(unittest.TestCase):
    def setUp(self):
        api._jobs.clear()
        api._analysis_jobs.clear()

    def test_health_reports_storage_paths(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_create_job_processes_upload_and_exposes_lesson(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"), patch.object(
                api, "_executor", ImmediateExecutor()
            ), patch.object(
                api, "DiarizationPipelineRunner", FakePipelineRunner
            ):
                with TestClient(api.app) as client:
                    created = client.post(
                        "/api/jobs",
                        data={"profile": "Spanish_Lessons", "language": "es", "model_size": "tiny"},
                        files={"audio": ("lesson.wav", b"fake wav bytes", "audio/wav")},
                    )
                    job = created.json()
                    job_detail = client.get(f"/api/jobs/{job['id']}").json()
                    lesson_id = job_detail["lesson_id"]
                    lessons = client.get("/api/lessons").json()
                    lesson = client.get(f"/api/lessons/{lesson_id}").json()

        self.assertEqual(created.status_code, 200)
        self.assertEqual(job_detail["status"], "succeeded")
        self.assertEqual(job_detail["progress"], 100)
        self.assertEqual(job_detail["result"]["lesson_id"], lesson_id)
        self.assertEqual(lessons["lessons"][0]["id"], lesson_id)
        self.assertEqual(lesson["meta"]["profile"], "Spanish_Lessons")
        self.assertEqual(lesson["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(lesson["transcript"], "SPEAKER_00: hola\n")

    def test_missing_job_returns_404(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.get("/api/jobs/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_update_lesson_speakers_persists_labels_and_student_speakers(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({"profile": "test"}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(
                json.dumps(
                    [
                        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hola"},
                        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "bien"},
                    ]
                ),
                encoding="utf-8",
            )
            (lesson_dir / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.patch(
                        "/api/lessons/lesson-1/speakers",
                        json={
                            "speaker_labels": {"SPEAKER_00": "Tutor", "SPEAKER_01": "Student"},
                            "student_speakers": ["SPEAKER_01"],
                        },
                    )

        self.assertEqual(response.status_code, 200)
        meta = response.json()["meta"]
        self.assertEqual(meta["speaker_labels"]["SPEAKER_00"], "Tutor")
        self.assertEqual(meta["student_speakers"], ["SPEAKER_01"])
        self.assertIn("speaker_reviewed_at", meta)

    def test_update_lesson_speakers_rejects_unknown_speaker(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(
                json.dumps([{"speaker": "SPEAKER_00", "text": "hola"}]),
                encoding="utf-8",
            )

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.patch(
                        "/api/lessons/lesson-1/speakers",
                        json={"student_speakers": ["SPEAKER_99"]},
                    )

        self.assertEqual(response.status_code, 400)

    def test_analyze_lesson_writes_ai_stats_and_updates_meta(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({"profile": "test"}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(
                json.dumps([{"speaker": "SPEAKER_00", "text": "hola"}]),
                encoding="utf-8",
            )
            (lesson_dir / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"), patch.object(
                api, "DiarizationPipelineRunner", FakePipelineRunner
            ):
                with TestClient(api.app) as client:
                    response = client.post(
                        "/api/lessons/lesson-1/analyze",
                        json={"provider": "ollama", "model": "gemma4:e4b"},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["ai_stats"]["grammar_score"], 88)
        self.assertEqual(body["meta"]["llm_provider"], "ollama")
        self.assertEqual(body["meta"]["llm_model"], "gemma4:e4b")
        self.assertIn("analyzed_at", body["meta"])

    def test_analyze_lesson_returns_pipeline_error_detail(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({"profile": "test"}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(
                json.dumps([{"speaker": "SPEAKER_00", "text": "hola"}]),
                encoding="utf-8",
            )
            (lesson_dir / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"), patch.object(
                api, "DiarizationPipelineRunner", FailingPipelineRunner
            ):
                with TestClient(api.app) as client:
                    response = client.post("/api/lessons/lesson-1/analyze")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Error: model not found")

    def test_create_analysis_job_runs_in_background_and_can_be_polled(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({"profile": "test"}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(
                json.dumps([{"speaker": "SPEAKER_00", "text": "hola"}]),
                encoding="utf-8",
            )
            (lesson_dir / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"), patch.object(
                api, "_executor", ImmediateExecutor()
            ), patch.object(
                api, "DiarizationPipelineRunner", FakePipelineRunner
            ):
                with TestClient(api.app) as client:
                    created = client.post(
                        "/api/lessons/lesson-1/analysis-jobs",
                        json={"provider": "ollama", "model": "gemma4:e4b"},
                    )
                    job = created.json()
                    polled = client.get(f"/api/analysis-jobs/{job['id']}").json()
                    lesson = client.get("/api/lessons/lesson-1").json()

        self.assertEqual(created.status_code, 200)
        self.assertEqual(polled["status"], "succeeded")
        self.assertEqual(polled["progress"], 100)
        self.assertEqual(polled["lesson_id"], "lesson-1")
        self.assertEqual(polled["result"]["ai_stats"]["grammar_score"], 88)
        self.assertEqual(lesson["ai_stats"]["llm_model"], "gemma4:e4b")

    def test_missing_analysis_job_returns_404(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.get("/api/analysis-jobs/does-not-exist")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
