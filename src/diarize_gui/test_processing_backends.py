import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from diarize_gui.pipeline import DiarizationPipelineRunner
from diarize_gui.processing_backends import (
    ASRRunResult,
    FluidAudioDiarizationBackend,
    _model_name_for_mlx,
    assign_speakers_by_overlap,
    clean_transcript_text,
    prepare_transcript_segments,
    resolve_asr_backend,
    resolve_diarization_backend,
    single_speaker_diarization_from_segments,
)


class ProcessingBackendTests(unittest.TestCase):
    def test_mlx_whisper_model_names_use_existing_hf_repos(self):
        self.assertEqual(_model_name_for_mlx("small"), "mlx-community/whisper-small-mlx")
        self.assertEqual(_model_name_for_mlx("whisper-small"), "mlx-community/whisper-small-mlx")
        self.assertEqual(_model_name_for_mlx("mlx-community/whisper-small-mlx"), "mlx-community/whisper-small-mlx")

    def test_auto_prefers_mlx_on_apple_silicon(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            def _fake_import(name):
                if name in {"mlx_whisper", "whisperx", "whisper"}:
                    return object()
                return None

            safe_import.side_effect = _fake_import
            backend, meta = resolve_asr_backend("auto")

        self.assertEqual(backend.name, "mlx")
        self.assertEqual(meta["selected"], "mlx")

    def test_mlx_can_still_be_selected_explicitly_on_apple_silicon(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            def _fake_import(name):
                if name == "mlx_whisper":
                    return object()
                return None

            safe_import.side_effect = _fake_import
            backend, meta = resolve_asr_backend("mlx")

        self.assertEqual(backend.name, "mlx")
        self.assertEqual(meta["selected"], "mlx")

    def test_unavailable_explicit_mlx_uses_next_apple_backend(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            safe_import.side_effect = lambda name: object() if name in {"whisper", "whisperx"} else None
            backend, meta = resolve_asr_backend("mlx")

        self.assertEqual(backend.name, "whisper_mps")
        self.assertEqual(meta["fallback"], "whisper_mps")

    def test_auto_prefers_openai_whisper_when_mlx_unavailable(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            def _fake_import(name):
                if name in {"whisperx", "whisper"}:
                    return object()
                return None

            safe_import.side_effect = _fake_import
            backend, meta = resolve_asr_backend("auto")

        self.assertEqual(backend.name, "whisper_mps")
        self.assertEqual(meta["selected"], "whisper_mps")

    def test_auto_falls_back_to_whisperx_when_apple_backends_are_unavailable(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends._safe_import"
        ) as safe_import:
            safe_import.side_effect = lambda name: object() if name == "whisperx" else None
            backend, meta = resolve_asr_backend("auto")

        self.assertEqual(backend.name, "whisperx")
        self.assertEqual(meta["selected"], "whisperx")

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

    def test_single_speaker_diarization_uses_transcript_timestamps(self):
        diarization = single_speaker_diarization_from_segments(
            [
                {"start": 1.0, "end": 2.5, "text": "hola"},
                {"start": 2.5, "end": 4.0, "text": "mundo"},
            ]
        )

        self.assertEqual(
            diarization,
            [
                {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_00"},
            ],
        )

    def test_diarization_auto_prefers_fluidaudio_on_apple(self):
        with patch("diarize_gui.processing_backends.is_apple_silicon", return_value=True), patch(
            "diarize_gui.processing_backends.find_fluidaudio_binary", return_value="/tmp/fluidaudiocli"
        ):
            backend, meta = resolve_diarization_backend("auto")

        self.assertEqual(backend.name, "fluidaudio")
        self.assertEqual(meta["selected"], "fluidaudio")

    def test_fluidaudio_output_is_normalized(self):
        payload = {
            "processingTimeSeconds": 1.2,
            "realTimeFactor": 250.0,
            "speakerCount": 2,
            "segments": [
                {"startTimeSeconds": 0.0, "endTimeSeconds": 1.5, "speakerId": "S1"},
                {"startTimeSeconds": 1.5, "endTimeSeconds": 3.0, "speakerId": "S2"},
            ],
        }

        def fake_run(command, **kwargs):
            output_path = command[command.index("--output") + 1]
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("diarize_gui.processing_backends.find_fluidaudio_binary", return_value="/tmp/fluid"), patch(
            "diarize_gui.processing_backends.subprocess.run", side_effect=fake_run
        ):
            segments, meta = FluidAudioDiarizationBackend().diarize("lesson.wav", num_speakers=2)

        self.assertEqual(segments[0], {"start": 0.0, "end": 1.5, "speaker": "S1"})
        self.assertEqual(meta["device"], "apple_neural_engine")
        self.assertEqual(meta["speaker_count"], 2)

    def test_pipeline_falls_back_when_diarization_backend_fails(self):
        class FakeAsrBackend:
            name = "fake_asr"

            def transcribe(self, audio_path, *, model_size, language=None, config=None):
                return ASRRunResult(
                    segments=[{"start": 0.0, "end": 1.0, "text": "hola"}],
                    language="es",
                    backend=self.name,
                    device="cpu",
                    compute_type="int8",
                    word_timestamps_available=False,
                    metadata={},
                )

        class FailingDiarizationBackend:
            name = "pyannote"

            def diarize(self, *args, **kwargs):
                raise FileNotFoundError("Offline Pyannote config not found")

        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            preprocess = SimpleNamespace(
                normalized_path=str(output_dir / "normalized.wav"),
                already_normalized=False,
                reused_cache=False,
                sample_rate=16000,
                channels=1,
            )
            with patch("diarize_gui.pipeline.preprocess_audio_mono_16k", return_value=preprocess), patch(
                "diarize_gui.pipeline.resolve_asr_backend",
                return_value=(FakeAsrBackend(), {"requested": "auto", "selected": "fake_asr", "fallback": None}),
            ), patch(
                "diarize_gui.pipeline.resolve_diarization_backend",
                return_value=(
                    FailingDiarizationBackend(),
                    {"requested": "pyannote", "selected": "pyannote", "fallback": None, "notes": []},
                ),
            ):
                runner = DiarizationPipelineRunner()
                _, json_path = runner.process_audio("lesson.wav", str(output_dir))

            self.assertEqual(runner.last_processing_meta["diarization_backend_fallback"], "single_speaker")
            self.assertEqual(runner.last_result["segments"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(runner.last_result["metadata"]["diarization"]["backend"], "single_speaker_fallback")
            self.assertTrue(Path(json_path).exists())


if __name__ == "__main__":
    unittest.main()
