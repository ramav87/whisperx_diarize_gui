import json
import tempfile
import unittest
from pathlib import Path

from diarize_gui.pipeline import DiarizationPipelineRunner


class CapturingAnalysisRunner(DiarizationPipelineRunner):
    def analyze_with_llm(self, **kwargs):
        self.captured_prompt = kwargs["user_prompt"]
        self.captured_text = kwargs["external_text"]
        return json.dumps(
            {
                "grammar_score": 8,
                "topics": ["Business", "Finance", "Vocabulary"],
                "golden_words": ["presupuestos (budgets)"],
                "corrections": 2,
                "feedback": "Good control with some article errors.",
                "topic_difficulty": 4,
                "idea_density": 4,
                "abstraction_level": 7,
                "cognitive_branching": 5,
                "technical_density": 3,
                "discourse_depth": 6,
                "lexical_retrieval_pressure": 5,
                "topic_tags": ["business"],
                "context_notes": "Mixed concrete and abstract discussion.",
                "self_repair_observations": "Some successful word searches.",
            }
        )


class PipelineAnalysisTests(unittest.TestCase):
    def test_load_lesson_falls_back_to_mirrored_audio(self):
        with tempfile.TemporaryDirectory() as root:
            lesson_dir = Path(root)
            (lesson_dir / "segments.json").write_text("[]", encoding="utf-8")
            (lesson_dir / "audio.wav").write_bytes(b"RIFF-test")
            (lesson_dir / "meta.json").write_text(
                json.dumps({"source_audio_path": "/server/path/that/does/not/exist.wav"}),
                encoding="utf-8",
            )

            runner = DiarizationPipelineRunner()
            runner.load_lesson_artifacts(str(lesson_dir))

            self.assertEqual(runner.last_audio_path, str(lesson_dir / "audio.wav"))

    def test_compute_ai_metrics_marks_student_scope_in_prompt_and_output(self):
        with tempfile.TemporaryDirectory() as root:
            lesson_dir = Path(root)
            (lesson_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "speaker_labels": {
                            "SPEAKER_00": "Tutor",
                            "SPEAKER_01": "Student",
                        },
                        "student_speakers": ["SPEAKER_01"],
                    }
                ),
                encoding="utf-8",
            )
            (lesson_dir / "segments.json").write_text(
                json.dumps(
                    [
                        {
                            "start": 0,
                            "end": 2,
                            "speaker": "SPEAKER_00",
                            "text": "Como se dice glow in the dark?",
                        },
                        {
                            "start": 2,
                            "end": 14,
                            "speaker": "SPEAKER_01",
                            "text": "Tiene un poder de lucifera.",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            runner = CapturingAnalysisRunner()
            success = runner.compute_ai_metrics(str(lesson_dir), mode="openai", model="test-model")
            stats = json.loads((lesson_dir / "ai_stats.json").read_text(encoding="utf-8"))

        self.assertTrue(success)
        self.assertIn("Evaluate only these learner/student speaker IDs: SPEAKER_01", runner.captured_prompt)
        self.assertIn("Tutor (SPEAKER_00): Como se dice glow in the dark?", runner.captured_text)
        self.assertIn("Student (SPEAKER_01): Tiene un poder de lucifera.", runner.captured_text)
        self.assertEqual(stats["grammar_score"], 80)
        self.assertEqual(stats["analysis_schema_version"], 2)
        self.assertEqual(stats["analysis_provenance"]["transcript_coverage"], 1.0)
        self.assertEqual(stats["analysis_scope"]["student_speakers"], ["SPEAKER_01"])
        self.assertEqual(stats["analysis_scope"]["speaker_labels"]["SPEAKER_01"], "Student")

    def test_invalid_model_output_is_not_saved_as_a_default_score(self):
        class InvalidRunner(DiarizationPipelineRunner):
            def analyze_with_llm(self, **kwargs):
                return "not valid JSON"

        with tempfile.TemporaryDirectory() as root:
            lesson_dir = Path(root)
            (lesson_dir / "meta.json").write_text(
                json.dumps({"student_speakers": ["SPEAKER_01"]}),
                encoding="utf-8",
            )
            (lesson_dir / "segments.json").write_text(
                json.dumps([
                    {"start": 0, "end": 12, "speaker": "SPEAKER_01", "text": "Hablo durante una lección."}
                ]),
                encoding="utf-8",
            )
            runner = InvalidRunner()
            success = runner.compute_ai_metrics(str(lesson_dir), mode="openai", model="test-model")

            self.assertFalse(success)
            self.assertFalse((lesson_dir / "ai_stats.json").exists())
            self.assertIn("Invalid structured AI metrics", runner.last_ai_metrics_error)


if __name__ == "__main__":
    unittest.main()
