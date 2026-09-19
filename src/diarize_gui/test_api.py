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
        (out / "segments_raw.json").write_text(
            json.dumps([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hola"}]),
            encoding="utf-8",
        )
        (out / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")
        (out / "transcript_raw.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")
        (out / "transcript_cleaned.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")
        (out / "transcript_cleaned_highlighted.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")
        (out / "analysis.txt").write_text("Good work.\n", encoding="utf-8")
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
        self._profiles_tmp = tempfile.TemporaryDirectory()
        self._profiles_patch = patch.object(api, "PROFILES_DIR", Path(self._profiles_tmp.name) / "profiles")
        self._profiles_patch.start()

    def tearDown(self):
        self._profiles_patch.stop()
        self._profiles_tmp.cleanup()

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
        self.assertIn("api_version", response.json())

    def test_capabilities_reports_contract_surfaces(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.get("/api/capabilities")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("transcript_cleaned", body["artifact_names"])
        self.assertIn("lesson_context", body["endpoints"])
        self.assertIn("list_profiles", body["endpoints"])
        self.assertIn("create_profile", body["endpoints"])
        self.assertIn("patch_profile", body["endpoints"])
        self.assertIn("profile_dashboard", body["endpoints"])
        self.assertIn("profile_lessons", body["endpoints"])
        self.assertIn("create_profile_processing_job", body["endpoints"])

    def test_profiles_can_be_created_listed_and_read(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    created = client.put(
                        "/api/profiles/Spanish%20Lessons",
                        json={
                            "display_name": "Spanish Lessons",
                            "settings": {"whisper_model_size": "large-v3", "language": "Spanish"},
                        },
                    )
                    listed = client.get("/api/profiles")
                    fetched = client.get("/api/profiles/Spanish_Lessons")

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["id"], "Spanish_Lessons")
        self.assertEqual(fetched.json()["settings"]["language"], "Spanish")
        self.assertEqual(listed.json()["profiles"][0]["id"], "Spanish_Lessons")

    def test_profiles_can_be_created_and_patched_as_server_resources(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    created = client.post(
                        "/api/profiles",
                        json={
                            "id": "Spanish Lessons",
                            "display_name": "Spanish Lessons",
                            "owner_id": None,
                            "settings": {"language": "Spanish", "whisper_model_size": "tiny"},
                        },
                    )
                    patched = client.patch(
                        "/api/profiles/Spanish_Lessons",
                        json={"settings": {"analysis_model": "gemma"}, "display_name": "Spanish"},
                    )
                    duplicate = client.post("/api/profiles", json={"id": "Spanish Lessons"})

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["id"], "Spanish_Lessons")
        self.assertEqual(created.json()["owner_id"], None)
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.json()["display_name"], "Spanish")
        self.assertEqual(patched.json()["settings"]["language"], "Spanish")
        self.assertEqual(patched.json()["settings"]["analysis_model"], "gemma")
        self.assertEqual(duplicate.status_code, 409)

    def test_profile_scoped_lessons_are_canonical(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lessons_root = data_dir / "lessons"
            for lesson_id, profile in (("lesson-1", "A"), ("lesson-2", "B")):
                lesson_dir = lessons_root / lesson_id
                lesson_dir.mkdir(parents=True)
                (lesson_dir / "meta.json").write_text(json.dumps({"profile": profile}), encoding="utf-8")
                (lesson_dir / "segments.json").write_text(json.dumps([]), encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", lessons_root):
                with TestClient(api.app) as client:
                    response = client.get("/api/profiles/A/lessons")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["profile"], "A")
        lessons = response.json()["lessons"]
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["id"], "lesson-1")

    def test_profile_scoped_processing_job_owns_profile(self):
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
                        "/api/profiles/Spanish_Lessons/jobs",
                        data={"language": "es", "model_size": "tiny"},
                        files={"audio": ("lesson.wav", b"fake wav bytes", "audio/wav")},
                    )
                    job = client.get(f"/api/jobs/{created.json()['id']}").json()
                    lesson = client.get(f"/api/lessons/{job['lesson_id']}").json()
                    profile_lessons = client.get("/api/profiles/Spanish_Lessons/lessons").json()

        self.assertEqual(created.status_code, 200)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(lesson["meta"]["profile"], "Spanish_Lessons")
        self.assertEqual(profile_lessons["lessons"][0]["id"], job["lesson_id"])

    def test_profile_dashboard_computes_server_side_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lessons_root = data_dir / "lessons"

            lesson_1 = lessons_root / "20260501_morning"
            lesson_1.mkdir(parents=True)
            (lesson_1 / "meta.json").write_text(
                json.dumps(
                    {
                        "profile": "Spanish_Lessons",
                        "recorded_at": "2026-05-01T09:00:00",
                        "duration_sec": 7200,
                        "student_speakers": ["SPEAKER_01"],
                    }
                ),
                encoding="utf-8",
            )
            (lesson_1 / "segments.json").write_text(
                json.dumps(
                    [
                        {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "text": "intro"},
                        {
                            "start": 32.0,
                            "end": 92.0,
                            "speaker": "SPEAKER_01",
                            "text": " ".join(["hola"] * 30),
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (lesson_1 / "ai_stats.json").write_text(
                json.dumps({"grammar_score": 70, "golden_words": ["entonces"], "topic_difficulty": 3}),
                encoding="utf-8",
            )

            lesson_2 = lessons_root / "20260503_evening"
            lesson_2.mkdir(parents=True)
            (lesson_2 / "meta.json").write_text(
                json.dumps(
                    {
                        "profile": "Spanish_Lessons",
                        "recorded_at": "2026-05-03T18:00:00",
                        "duration_sec": 3600,
                        "student_speakers": ["SPEAKER_01"],
                    }
                ),
                encoding="utf-8",
            )
            (lesson_2 / "segments.json").write_text(
                json.dumps(
                    [
                        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00", "text": "intro"},
                        {
                            "start": 12.0,
                            "end": 72.0,
                            "speaker": "SPEAKER_01",
                            "text": " ".join(["bien"] * 60),
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (lesson_2 / "ai_stats.json").write_text(
                json.dumps({"grammar_score": 90, "golden_words": ["aunque"], "topic_difficulty": 5}),
                encoding="utf-8",
            )

            other_lesson = lessons_root / "20260504_other"
            other_lesson.mkdir(parents=True)
            (other_lesson / "meta.json").write_text(
                json.dumps({"profile": "Other", "recorded_at": "2026-05-04T09:00:00", "duration_sec": 999}),
                encoding="utf-8",
            )
            (other_lesson / "segments.json").write_text(json.dumps([]), encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", lessons_root):
                with TestClient(api.app) as client:
                    response = client.get("/api/profiles/Spanish_Lessons/dashboard")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        summary = body["summary"]
        self.assertEqual(body["profile"], "Spanish_Lessons")
        self.assertEqual(summary["lesson_count"], 2)
        self.assertEqual(summary["analyzed_lesson_count"], 2)
        self.assertEqual(summary["total_recording_sec"], 10800)
        self.assertEqual(summary["student_total_words"], 90)
        self.assertEqual(summary["average_grammar_score"], 80)
        self.assertEqual(summary["golden_words"][:2], ["aunque", "entonces"])
        self.assertEqual(len(body["trends"]["grammar"]), 2)
        self.assertEqual(len(body["trends"]["fluency"]), 2)
        self.assertEqual(body["trends"]["context"][1]["context_metrics"]["practice_hours_last_7_days"], 2.0)
        self.assertEqual(summary["automaticity_gap"]["warm_session_count"], 1)
        self.assertEqual(summary["automaticity_gap"]["cold_session_count"], 1)
        self.assertEqual(summary["automaticity_gap"]["automaticity_gap"], 20)

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
        self.assertTrue(any(item["name"] == "transcript" for item in lesson["artifacts"]))

    def test_list_lessons_can_filter_by_profile(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lessons_root = data_dir / "lessons"
            for lesson_id, profile in (("lesson-1", "A"), ("lesson-2", "B")):
                lesson_dir = lessons_root / lesson_id
                lesson_dir.mkdir(parents=True)
                (lesson_dir / "meta.json").write_text(json.dumps({"profile": profile}), encoding="utf-8")
                (lesson_dir / "segments.json").write_text(json.dumps([]), encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", lessons_root):
                with TestClient(api.app) as client:
                    response = client.get("/api/lessons?profile=A")

        self.assertEqual(response.status_code, 200)
        lessons = response.json()["lessons"]
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["profile"], "A")

    def test_processing_job_state_is_durable(self):
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
                        data={"profile": "Spanish_Lessons"},
                        files={"audio": ("lesson.wav", b"fake wav bytes", "audio/wav")},
                    )
                    job_id = created.json()["id"]

                api._jobs.clear()
                api._load_jobs_from_disk()
                reloaded = api._jobs[job_id]

        self.assertEqual(reloaded.status, "succeeded")
        self.assertEqual(reloaded.progress, 100)

    def test_interrupted_job_state_is_marked_failed_on_reload(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            job_dir = data_dir / "jobs" / "processing"
            job_dir.mkdir(parents=True)
            (job_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "status": "running",
                        "created_at": "2026-05-25T12:00:00",
                        "updated_at": "2026-05-25T12:00:00",
                        "progress": 25,
                        "message": "Working",
                        "result": {},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(api, "DATA_DIR", data_dir):
                api._load_jobs_from_disk()
                state = api._jobs["job-1"]

        self.assertEqual(state.status, "failed")
        self.assertIn("Server restarted", state.error)

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

    def test_lesson_artifacts_can_be_listed_and_downloaded(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({"profile": "test"}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(json.dumps([]), encoding="utf-8")
            (lesson_dir / "transcript.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")
            (lesson_dir / "transcript_cleaned.txt").write_text("SPEAKER_00: hola\n", encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    listed = client.get("/api/lessons/lesson-1/artifacts")
                    downloaded = client.get("/api/lessons/lesson-1/artifacts/transcript")
                    exported = client.get("/api/lessons/lesson-1/exports/transcript?variant=cleaned&format=txt")

        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(item["name"] == "transcript" for item in listed.json()["artifacts"]))
        self.assertEqual(downloaded.text, "SPEAKER_00: hola\n")
        self.assertEqual(exported.text, "SPEAKER_00: hola\n")

    def test_analysis_export_supports_text_and_json(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(json.dumps({"profile": "test"}), encoding="utf-8")
            (lesson_dir / "segments.json").write_text(json.dumps([]), encoding="utf-8")
            (lesson_dir / "analysis.txt").write_text("Nice analysis.\n", encoding="utf-8")
            (lesson_dir / "ai_stats.json").write_text(json.dumps({"grammar_score": 88}), encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    text_response = client.get("/api/lessons/lesson-1/exports/analysis")
                    json_response = client.get("/api/lessons/lesson-1/exports/analysis?format=json")

        self.assertEqual(text_response.text, "Nice analysis.\n")
        self.assertEqual(json_response.json()["grammar_score"], 88)

    def test_context_metrics_can_be_recomputed_and_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            lesson_dir = data_dir / "lessons" / "lesson-1"
            lesson_dir.mkdir(parents=True)
            (lesson_dir / "meta.json").write_text(
                json.dumps({"profile": "test", "student_speakers": ["SPEAKER_01"]}),
                encoding="utf-8",
            )
            (lesson_dir / "segments.json").write_text(
                json.dumps(
                    [
                        {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_01", "text": "uno dos tres cuatro"},
                    ]
                ),
                encoding="utf-8",
            )
            (lesson_dir / "ai_stats.json").write_text(json.dumps({"grammar_score": 70}), encoding="utf-8")

            with patch.object(api, "DATA_DIR", data_dir), patch.object(
                api, "UPLOADS_DIR", data_dir / "uploads"
            ), patch.object(api, "LESSONS_DIR", data_dir / "lessons"):
                with TestClient(api.app) as client:
                    response = client.patch(
                        "/api/lessons/lesson-1/context",
                        json={
                            "practice_hours_last_7_days": 0.5,
                            "topic_difficulty": 5,
                            "idea_density": 4,
                            "notes": "Hard topic",
                        },
                    )
                    fetched = client.get("/api/lessons/lesson-1/context")

        self.assertEqual(response.status_code, 200)
        context = response.json()["context_metrics"]
        self.assertEqual(context["topic_difficulty"], 5.0)
        self.assertEqual(context["raw_grammar_score"], 70.0)
        self.assertIn("interpretation", fetched.json())

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
