import unittest
from unittest.mock import patch

from diarize_gui.processing_backends import (
    assign_speakers_by_overlap,
    clean_transcript_text,
    prepare_transcript_segments,
    resolve_asr_backend,
)


class ProcessingBackendTests(unittest.TestCase):
    def test_auto_prefers_mlx_on_apple_silicon(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            def _fake_import(name):
                if name == "mlx_whisper":
                    return object()
                return None

            safe_import.side_effect = _fake_import
            backend, meta = resolve_asr_backend("auto")

        self.assertEqual(backend.name, "mlx")
        self.assertEqual(meta["selected"], "mlx")

    def test_auto_prefers_whisperx_on_non_apple(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=False), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            def _fake_import(name):
                if name == "whisperx":
                    return object()
                return None

            safe_import.side_effect = _fake_import
            backend, meta = resolve_asr_backend("auto")

        self.assertEqual(backend.name, "whisperx")
        self.assertEqual(meta["selected"], "whisperx")

    def test_overlap_assignment_prefers_best_matching_speaker(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 2.0, "end": 4.0, "text": "world"},
        ]
        diarization = [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 4.0, "speaker": "SPEAKER_01"},
        ]

        assigned = assign_speakers_by_overlap(segments, diarization)

        self.assertEqual(assigned[0]["speaker"], "SPEAKER_00")
        self.assertEqual(assigned[1]["speaker"], "SPEAKER_01")

    def test_cleanup_removes_simple_repetitions_without_paraphrasing(self):
        cleaned, meta = clean_transcript_text("hola hola yo yo quiero quiero ir")
        self.assertEqual(cleaned, "hola yo quiero ir")
        self.assertTrue(meta["changed"])

    def test_prepare_segments_adds_confidence_flags(self):
        cleaned, raw = prepare_transcript_segments(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "eh eh hola hola",
                    "avg_logprob": -1.2,
                    "no_speech_prob": 0.7,
                }
            ]
        )
        self.assertEqual(raw[0]["text"], "eh eh hola hola")
        self.assertEqual(cleaned[0]["text"], "eh hola")
        self.assertIn("confidence", cleaned[0])
        self.assertTrue(cleaned[0]["low_confidence"])


if __name__ == "__main__":
    unittest.main()
