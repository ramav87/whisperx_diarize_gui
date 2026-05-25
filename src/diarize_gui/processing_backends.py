from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .utils import detect_device, is_apple_silicon


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _backend_status(config: Optional[dict], text: str, progress: Optional[float] = None) -> None:
    config = config or {}
    status_callback = config.get("status_callback")
    progress_callback = config.get("progress_callback")
    if status_callback:
        status_callback(text)
    if progress is not None and progress_callback:
        progress_callback(progress)


def _model_name_for_mlx(model_size: str) -> str:
    size = (model_size or "small").strip()
    if size.startswith("mlx-community/"):
        return size
    if size.startswith("whisper-"):
        return f"mlx-community/{size}"
    return f"mlx-community/whisper-{size}"


FILLER_WORDS = {
    "eh",
    "em",
    "mm",
    "hm",
    "uh",
    "um",
    "este",
    "estee",
    "pues",
}


def _segment_dict(
    start: Any,
    end: Any,
    text: str,
    speaker: Optional[str] = None,
    words=None,
    extra: Optional[dict] = None,
) -> dict:
    seg = {
        "start": float(start or 0.0),
        "end": float(end or 0.0),
        "text": (text or "").strip(),
    }
    if speaker:
        seg["speaker"] = speaker
    if words:
        seg["words"] = words
    if extra:
        seg.update(extra)
    return seg


def _normalize_segments(raw_segments: List[dict]) -> List[dict]:
    normalized = []
    for seg in raw_segments or []:
        if not isinstance(seg, dict):
            continue
        normalized.append(
            _segment_dict(
                seg.get("start"),
                seg.get("end"),
                seg.get("text", ""),
                speaker=seg.get("speaker"),
                words=seg.get("words"),
                extra={k: v for k, v in seg.items() if k not in {"start", "end", "text", "speaker", "words"}},
            )
        )
    return normalized


def _collapse_repeated_phrases(tokens: List[str], max_phrase_len: int = 3) -> List[str]:
    if len(tokens) < 2:
        return tokens

    changed = True
    while changed:
        changed = False
        for size in range(min(max_phrase_len, len(tokens) // 2), 0, -1):
            out: List[str] = []
            i = 0
            while i < len(tokens):
                a = tokens[i : i + size]
                b = tokens[i + size : i + 2 * size]
                if len(a) == size and len(b) == size and a and a == b:
                    out.extend(a)
                    i += size * 2
                    changed = True
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
    return tokens


def clean_transcript_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Light-touch cleanup for ASR output:
    - whitespace normalization
    - repeated filler removal
    - obvious token/phrase de-duplication
    """
    raw = (text or "").strip()
    if not raw:
        return "", {"changed": False, "removed_tokens": 0}

    normalized = re.sub(r"\s+", " ", raw)
    tokens = normalized.split(" ")
    cleaned_tokens: List[str] = []
    removed = 0
    for tok in tokens:
        tok_norm = tok.strip(".,;:!?¿¡()[]{}\"'").lower()
        if cleaned_tokens:
            prev_norm = cleaned_tokens[-1].strip(".,;:!?¿¡()[]{}\"'").lower()
            if tok_norm and tok_norm == prev_norm:
                removed += 1
                continue
            if tok_norm in FILLER_WORDS and prev_norm in FILLER_WORDS:
                removed += 1
                continue
        cleaned_tokens.append(tok)

    cleaned_tokens = _collapse_repeated_phrases(cleaned_tokens)
    cleaned = " ".join(cleaned_tokens)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    changed = cleaned != raw
    return cleaned, {"changed": changed, "removed_tokens": removed}


def _score_segment_confidence(seg: dict, raw_text: str, cleaned_text: str) -> Tuple[float, List[str]]:
    score = 0.85
    reasons: List[str] = []

    avg_logprob = seg.get("avg_logprob")
    if isinstance(avg_logprob, (int, float)):
        score += max(-0.45, min(0.15, float(avg_logprob) / 4.0))
        if avg_logprob < -1.0:
            reasons.append("low avg_logprob")

    no_speech_prob = seg.get("no_speech_prob")
    if isinstance(no_speech_prob, (int, float)):
        score -= min(0.4, float(no_speech_prob) * 0.5)
        if no_speech_prob > 0.6:
            reasons.append("high no_speech_prob")

    compression_ratio = seg.get("compression_ratio")
    if isinstance(compression_ratio, (int, float)) and compression_ratio > 2.2:
        score -= 0.1
        reasons.append("high compression_ratio")

    raw_norm = re.sub(r"\s+", " ", raw_text or "").strip().lower()
    cleaned_norm = re.sub(r"\s+", " ", cleaned_text or "").strip().lower()
    if raw_norm and raw_norm != cleaned_norm:
        score -= 0.04

    if len(cleaned_norm.split()) <= 2:
        score -= 0.08

    score = max(0.0, min(1.0, score))
    if score < 0.65:
        reasons.append("low confidence")
    return score, reasons


def prepare_transcript_segments(raw_segments: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Returns (cleaned_segments, raw_segments_normalized).
    """
    raw_normalized = _normalize_segments(raw_segments)
    cleaned: List[dict] = []
    for seg in raw_normalized:
        raw_text = (seg.get("text") or "").strip()
        cleaned_text, cleanup_meta = clean_transcript_text(raw_text)
        confidence, reasons = _score_segment_confidence(seg, raw_text, cleaned_text)

        cleaned.append(
            _segment_dict(
                seg.get("start"),
                seg.get("end"),
                cleaned_text,
                speaker=seg.get("speaker"),
                words=seg.get("words"),
                extra={
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text,
                    "cleanup_applied": bool(cleanup_meta.get("changed")),
                    "cleanup_removed_tokens": cleanup_meta.get("removed_tokens", 0),
                    "confidence": round(confidence, 3),
                    "low_confidence": confidence < 0.65,
                    "confidence_reasons": reasons,
                    "avg_logprob": seg.get("avg_logprob"),
                    "no_speech_prob": seg.get("no_speech_prob"),
                    "compression_ratio": seg.get("compression_ratio"),
                },
            )
        )

    return cleaned, raw_normalized


def assign_speakers_by_overlap(
    segments: List[dict],
    diarization_segments: List[dict],
) -> List[dict]:
    """
    Assigns each transcript segment the speaker label with the highest time overlap.
    """
    diarized = []
    for item in diarization_segments or []:
        try:
            diarized.append(
                {
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "speaker": str(item.get("speaker", "UNKNOWN")),
                }
            )
        except Exception:
            continue

    if not diarized:
        return _normalize_segments(segments)

    assigned = []
    for seg in segments or []:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except Exception:
            seg_start, seg_end = 0.0, 0.0

        best_speaker = seg.get("speaker") or "UNKNOWN"
        best_overlap = 0.0

        for dseg in diarized:
            overlap = max(0.0, min(seg_end, dseg["end"]) - max(seg_start, dseg["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        updated = dict(seg)
        updated["speaker"] = best_speaker
        assigned.append(
            _segment_dict(
                updated.get("start"),
                updated.get("end"),
                updated.get("text", ""),
                speaker=best_speaker,
                words=updated.get("words"),
                extra={
                    k: v
                    for k, v in updated.items()
                    if k not in {"start", "end", "text", "speaker", "words"}
                },
            )
        )

    return assigned


@dataclass
class ASRRunResult:
    segments: List[dict]
    language: Optional[str]
    backend: str
    device: str
    compute_type: Optional[str]
    word_timestamps_available: bool
    metadata: Dict[str, Any]


class ASRBackendBase:
    name = "base"

    def transcribe(
        self,
        audio_path: str,
        *,
        model_size: str,
        language: Optional[str],
        config: Optional[dict] = None,
    ) -> ASRRunResult:
        raise NotImplementedError


class WhisperXBackend(ASRBackendBase):
    name = "whisperx"

    def transcribe(
        self,
        audio_path: str,
        *,
        model_size: str,
        language: Optional[str],
        config: Optional[dict] = None,
    ) -> ASRRunResult:
        whisperx = _safe_import("whisperx")
        if whisperx is None:
            raise RuntimeError("WhisperX is not installed.")

        device = (config or {}).get("device") or detect_device()
        compute_type = (config or {}).get("compute_type") or ("int8" if device == "cpu" else "float16")
        batch_size = (config or {}).get("batch_size")

        _backend_status(config, f"Loading WhisperX model ({model_size}, {device}/{compute_type})...", 20)
        model = whisperx.load_model(model_size, device=device, compute_type=compute_type)
        _backend_status(config, "Loading normalized audio for ASR...", 28)
        audio = whisperx.load_audio(audio_path)
        transcribe_kwargs = {"language": language, "task": "transcribe"}
        if batch_size is not None:
            transcribe_kwargs["batch_size"] = int(batch_size)

        _backend_status(config, f"Transcribing audio with WhisperX (batch {batch_size or 'default'})...", 35)
        try:
            result = model.transcribe(audio, **transcribe_kwargs)
        except TypeError:
            transcribe_kwargs.pop("batch_size", None)
            result = model.transcribe(audio, **transcribe_kwargs)

        _backend_status(config, "Loading word-alignment model...", 72)
        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language") or language,
            device=device,
        )
        _backend_status(config, "Aligning transcript word timestamps...", 78)
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        return ASRRunResult(
            segments=_normalize_segments(aligned.get("segments", [])),
            language=aligned.get("language") or result.get("language") or language,
            backend=self.name,
            device=device,
            compute_type=compute_type,
            word_timestamps_available=True,
            metadata={
                "word_timestamps_available": True,
                "alignment_backend": "whisperx",
            },
        )


class WhisperMPBackend(ASRBackendBase):
    name = "whisper_mps"

    def transcribe(
        self,
        audio_path: str,
        *,
        model_size: str,
        language: Optional[str],
        config: Optional[dict] = None,
    ) -> ASRRunResult:
        torch = _safe_import("torch")
        whisper = _safe_import("whisper")
        if whisper is None:
            raise RuntimeError("openai-whisper is not installed.")

        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        _backend_status(config, f"Loading openai-whisper model ({model_size}, {device})...", 20)
        model = whisper.load_model(model_size, device=device)
        _backend_status(config, f"Transcribing audio with openai-whisper ({device})...", 35)
        result = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            fp16=(device != "cpu"),
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                _segment_dict(seg.get("start"), seg.get("end"), seg.get("text", ""), speaker=seg.get("speaker"))
            )

        return ASRRunResult(
            segments=segments,
            language=result.get("language") or language,
            backend=self.name,
            device=device,
            compute_type="fp16" if device == "mps" else "fp32",
            word_timestamps_available=False,
            metadata={
                "word_timestamps_available": False,
                "alignment_backend": None,
                "limitation": "openai-whisper on MPS preserves segment timestamps but does not provide word timestamps in this app path.",
            },
        )


class MLXWhisperBackend(ASRBackendBase):
    name = "mlx"

    def transcribe(
        self,
        audio_path: str,
        *,
        model_size: str,
        language: Optional[str],
        config: Optional[dict] = None,
    ) -> ASRRunResult:
        mlx_whisper = _safe_import("mlx_whisper")
        if mlx_whisper is None:
            raise RuntimeError("MLX Whisper is not installed.")

        model_ref = _model_name_for_mlx(model_size)
        kwargs: Dict[str, Any] = {"path_or_hf_repo": model_ref}
        if language:
            kwargs["language"] = language

        _backend_status(config, f"Loading/transcribing with MLX Whisper ({model_ref})...", 25)
        try:
            result = mlx_whisper.transcribe(audio_path, word_timestamps=True, **kwargs)
        except TypeError:
            result = mlx_whisper.transcribe(audio_path, **kwargs)

        segments = []
        for seg in result.get("segments", []):
            words = seg.get("words")
            segments.append(
                _segment_dict(
                    seg.get("start"),
                    seg.get("end"),
                    seg.get("text", ""),
                    speaker=seg.get("speaker"),
                    words=words,
                )
            )

        metadata = {
            "word_timestamps_available": any(bool(s.get("words")) for s in segments),
            "alignment_backend": "mlx_whisper" if any(bool(s.get("words")) for s in segments) else None,
            "model_ref": model_ref,
        }

        return ASRRunResult(
            segments=segments,
            language=result.get("language") or language,
            backend=self.name,
            device="mlx",
            compute_type="mlx",
            word_timestamps_available=metadata["word_timestamps_available"],
            metadata=metadata,
        )


def resolve_asr_backend(preferred: str) -> Tuple[ASRBackendBase, Dict[str, Any]]:
    """
    Returns (backend_instance, resolution_meta).
    """
    preferred = (preferred or "auto").strip().lower()
    resolution: Dict[str, Any] = {
        "requested": preferred,
        "selected": None,
        "fallback": None,
        "notes": [],
    }

    on_apple = is_apple_silicon()

    def _available(name: str) -> bool:
        if name == "mlx":
            return _safe_import("mlx_whisper") is not None
        if name == "whisper_mps":
            return _safe_import("whisper") is not None
        if name == "whisperx":
            return _safe_import("whisperx") is not None
        return False

    if preferred == "auto":
        if on_apple:
            for candidate in ("mlx", "whisper_mps", "whisperx"):
                if _available(candidate):
                    preferred = candidate
                    break
        else:
            preferred = "whisperx"

    backend_map = {
        "mlx": MLXWhisperBackend(),
        "whisper_mps": WhisperMPBackend(),
        "whisperx": WhisperXBackend(),
    }

    if preferred not in backend_map:
        resolution["notes"].append(f"Unknown backend '{preferred}', falling back to whisperx.")
        preferred = "whisperx"

    if not _available(preferred):
        if on_apple and preferred == "mlx" and _available("whisper_mps"):
            resolution["fallback"] = "whisper_mps"
            resolution["notes"].append("MLX Whisper was unavailable, so the app fell back to openai-whisper on MPS/CPU.")
            preferred = "whisper_mps"
        elif _available("whisperx"):
            resolution["fallback"] = "whisperx"
            resolution["notes"].append("Requested backend was unavailable, so the app fell back to WhisperX.")
            preferred = "whisperx"
        else:
            raise RuntimeError("No supported ASR backend is installed.")

    resolution["selected"] = preferred
    return backend_map[preferred], resolution


class DiarizationBackendBase:
    name = "base"

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        raise NotImplementedError


class PyannoteDiarizationBackend(DiarizationBackendBase):
    name = "pyannote"

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        from .pyannote_offline_loader import load_pyannote_pipeline

        pipeline = load_pyannote_pipeline()
        kwargs: Dict[str, Any] = {}
        if num_speakers:
            kwargs["num_speakers"] = int(num_speakers)
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

        try:
            annotation = pipeline(audio_path, **kwargs)
        except TypeError:
            annotation = pipeline(audio_path)

        segments = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": speaker,
                }
            )

        return segments, {
            "backend": self.name,
            "device": "cpu",
            "notes": [
                "Pyannote diarization runs as a separate CPU stage in the default path.",
            ],
        }


class SpeakerKitFutureBackend(DiarizationBackendBase):
    name = "speakerkit_future"

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        raise NotImplementedError(
            "SpeakerKit integration is not enabled yet. Use pyannote for now and plug in a future Apple-native backend later."
        )


def resolve_diarization_backend(preferred: str) -> Tuple[DiarizationBackendBase, Dict[str, Any]]:
    preferred = (preferred or "pyannote").strip().lower()
    meta = {"requested": preferred, "selected": None, "fallback": None, "notes": []}

    if preferred == "speakerkit_future":
        meta["notes"].append("SpeakerKit is reserved as a future Apple-native diarization hook.")
        meta["fallback"] = "pyannote"
        preferred = "pyannote"

    backend_map = {
        "pyannote": PyannoteDiarizationBackend(),
        "speakerkit_future": SpeakerKitFutureBackend(),
    }

    if preferred not in backend_map:
        meta["notes"].append(f"Unknown diarization backend '{preferred}', falling back to pyannote.")
        preferred = "pyannote"

    meta["selected"] = preferred
    return backend_map[preferred], meta
