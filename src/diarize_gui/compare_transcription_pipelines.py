from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import soundfile as sf
import numpy as np

from .audio_tools import preprocess_audio_mono_16k


DEFAULT_GEMMA_MODEL = "google/gemma-3n-E4B-it"
DEFAULT_GEMMA_PROMPT = (
    "Transcribe the speech in this audio clip exactly. "
    "Return only the transcript text. Do not add timestamps, commentary, or speaker labels."
)


@dataclass
class TimedResult:
    name: str
    seconds: float
    result: Dict[str, Any]


def _status(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _segment_text(segments: Iterable[dict]) -> str:
    return "\n".join((seg.get("text") or "").strip() for seg in segments if (seg.get("text") or "").strip())


def _normalize_for_compare(text: str) -> List[str]:
    text = re.sub(r"[^\w\s']", " ", (text or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def _levenshtein_distance(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        current = [i]
        for j, token_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def compare_texts(reference: str, candidate: str) -> Dict[str, Any]:
    ref_words = _normalize_for_compare(reference)
    cand_words = _normalize_for_compare(candidate)
    distance = _levenshtein_distance(ref_words, cand_words)
    denominator = max(1, len(ref_words))
    return {
        "reference_words": len(ref_words),
        "candidate_words": len(cand_words),
        "word_distance": distance,
        "word_error_rate_vs_existing": round(distance / denominator, 4),
        "sequence_similarity": round(difflib.SequenceMatcher(None, reference, candidate).ratio(), 4),
    }


def run_existing_asr(
    audio_path: str,
    *,
    backend: str,
    model_size: str,
    language: Optional[str],
    batch_size: Optional[int],
) -> Dict[str, Any]:
    from .processing_backends import prepare_transcript_segments, resolve_asr_backend

    asr_backend, resolution = resolve_asr_backend(backend)
    config = {
        "batch_size": batch_size,
        "status_callback": lambda text: _status(f"existing: {text}"),
    }
    asr_result = asr_backend.transcribe(audio_path, model_size=model_size, language=language, config=config)
    cleaned_segments, raw_segments = prepare_transcript_segments(asr_result.segments)
    return {
        "segments": cleaned_segments,
        "raw_segments": raw_segments,
        "text": _segment_text(cleaned_segments),
        "language": asr_result.language,
        "metadata": {
            "backend": asr_result.backend,
            "requested_backend": resolution["requested"],
            "selected_backend": resolution["selected"],
            "fallback_backend": resolution.get("fallback"),
            "model_size": model_size,
            "device": asr_result.device,
            "compute_type": asr_result.compute_type,
            "word_timestamps_available": asr_result.word_timestamps_available,
            "notes": asr_result.metadata,
        },
    }


def _write_pcm16_wav(path: str, data: Any, sample_rate: int) -> None:
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _audio_duration(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def _clip_audio(
    audio_path: str,
    *,
    start_seconds: float,
    max_seconds: Optional[float],
    output_dir: str,
) -> Dict[str, Any]:
    if start_seconds < 0:
        raise ValueError("--clip-start-seconds must be 0 or greater.")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("--clip-max-seconds must be greater than 0.")

    source_duration = _audio_duration(audio_path)
    if start_seconds >= source_duration:
        raise ValueError(
            f"--clip-start-seconds ({start_seconds}) is beyond audio duration ({source_duration:.3f})."
        )

    clip_duration = source_duration - start_seconds
    if max_seconds is not None:
        clip_duration = min(clip_duration, max_seconds)

    if start_seconds == 0 and max_seconds is None:
        return {
            "path": audio_path,
            "source_duration_seconds": source_duration,
            "start_seconds": 0.0,
            "duration_seconds": source_duration,
            "end_seconds": source_duration,
            "clipped": False,
        }

    info = sf.info(audio_path)
    sample_rate = int(info.samplerate)
    start_frame = int(round(start_seconds * sample_rate))
    frames_to_read = int(round(clip_duration * sample_rate))
    clip_path = os.path.join(output_dir, "comparison_clip.wav")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(audio_path) as source:
        source.seek(start_frame)
        data = source.read(frames_to_read, dtype="float32", always_2d=False)

    actual_duration = len(data) / float(sample_rate)
    _write_pcm16_wav(clip_path, data, sample_rate)
    return {
        "path": clip_path,
        "source_duration_seconds": source_duration,
        "start_seconds": float(start_seconds),
        "duration_seconds": actual_duration,
        "end_seconds": float(start_seconds) + actual_duration,
        "clipped": True,
    }


def _offset_words(words: Any, offset_seconds: float) -> Any:
    if not isinstance(words, list):
        return words
    shifted = []
    for word in words:
        if not isinstance(word, dict):
            shifted.append(word)
            continue
        updated = dict(word)
        for key in ("start", "end"):
            if isinstance(updated.get(key), (int, float)):
                updated[key] = float(updated[key]) + offset_seconds
        shifted.append(updated)
    return shifted


def _offset_segments(segments: Iterable[dict], offset_seconds: float) -> List[dict]:
    if not offset_seconds:
        return [dict(seg) for seg in segments or []]

    shifted = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        updated = dict(seg)
        for key in ("start", "end"):
            if isinstance(updated.get(key), (int, float)):
                updated[key] = float(updated[key]) + offset_seconds
        if "words" in updated:
            updated["words"] = _offset_words(updated.get("words"), offset_seconds)
        shifted.append(updated)
    return shifted


def _chunk_audio(audio_path: str, chunk_seconds: float, chunk_dir: str) -> List[Dict[str, Any]]:
    info = sf.info(audio_path)
    sample_rate = int(info.samplerate)
    total_frames = int(info.frames)
    chunk_frames = max(1, int(chunk_seconds * sample_rate))
    chunks: List[Dict[str, Any]] = []

    with sf.SoundFile(audio_path) as source:
        index = 0
        start_frame = 0
        while start_frame < total_frames:
            source.seek(start_frame)
            frames_to_read = min(chunk_frames, total_frames - start_frame)
            data = source.read(frames_to_read, dtype="float32", always_2d=False)
            start = start_frame / sample_rate
            end = (start_frame + frames_to_read) / sample_rate
            chunk_path = os.path.join(chunk_dir, f"chunk_{index:04d}_{start:.2f}_{end:.2f}.wav")
            _write_pcm16_wav(chunk_path, data, sample_rate)
            chunks.append({"index": index, "path": chunk_path, "start": start, "end": end})
            index += 1
            start_frame += frames_to_read

    return chunks


def _chunk_duration(chunks: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for chunk in chunks:
        try:
            total += max(0.0, float(chunk["end"]) - float(chunk["start"]))
        except Exception:
            continue
    return total


class GemmaAudioTranscriber:
    def __init__(
        self,
        model_id: str,
        *,
        prompt: str,
        max_new_tokens: int,
        device: str,
        torch_dtype: str,
        local_files_only: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Gemma transcription requires optional dependencies: transformers, torch, and any model-specific audio deps."
            ) from exc

        self.torch = torch
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" and hasattr(torch, torch_dtype) else "auto"
        load_kwargs: Dict[str, Any] = {"dtype": dtype, "local_files_only": local_files_only}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

        if device != "auto":
            self.model.to(device)
            self.device = device
        else:
            self.device = getattr(self.model, "device", None)

    def transcribe_chunk(self, chunk_path: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": chunk_path},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeError(
                "The selected processor did not accept an audio chat prompt. "
                "Use an audio-capable Gemma checkpoint, for example a Gemma 3n audio model."
            ) from exc

        if self.device not in (None, "auto"):
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, prompt_tokens:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return re.sub(r"\s+", " ", text).strip()


def run_gemma_asr(
    audio_path: str,
    *,
    model_id: str,
    chunk_seconds: float,
    prompt: str,
    max_new_tokens: int,
    device: str,
    torch_dtype: str,
    chunk_root: Optional[str] = None,
    start_chunk: int = 0,
    max_chunks: Optional[int] = None,
    list_chunks: bool = False,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    from .processing_backends import prepare_transcript_segments

    if chunk_seconds > 30:
        raise ValueError("Gemma audio chunks must be 30 seconds or shorter.")
    if start_chunk < 0:
        raise ValueError("--gemma-start-chunk must be 0 or greater.")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("--gemma-max-chunks must be 1 or greater.")

    if chunk_root:
        Path(chunk_root).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="diarize-gemma-chunks-", dir=chunk_root) as chunk_dir:
        all_chunks = _chunk_audio(audio_path, chunk_seconds, chunk_dir)
        chunks = all_chunks[start_chunk:]
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        selected_chunk_meta = [
            {
                "index": chunk["index"],
                "start": chunk["start"],
                "end": chunk["end"],
            }
            for chunk in chunks
        ]
        selected_duration = _chunk_duration(chunks)

        if list_chunks:
            return {
                "segments": [],
                "raw_segments": [],
                "text": "",
                "language": None,
                "metadata": {
                    "backend": "gemma_audio_transformers",
                    "model_id": model_id,
                    "chunk_seconds": chunk_seconds,
                    "available_chunks": len(all_chunks),
                    "processed_audio_seconds": round(selected_duration, 3),
                    "selected_chunks": [
                        {
                            "index": chunk["index"],
                            "start": chunk["start"],
                            "end": chunk["end"],
                            "path": chunk["path"],
                        }
                        for chunk in chunks
                    ],
                    "dry_run": True,
                },
            }

        if not chunks:
            raise ValueError(
                f"No chunks selected. Audio produced {len(all_chunks)} chunks; "
                f"--gemma-start-chunk was {start_chunk}."
            )

        _status(f"gemma: loading {model_id}")
        transcriber = GemmaAudioTranscriber(
            model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

        segments = []
        for selected_index, chunk in enumerate(chunks, 1):
            _status(
                f"gemma: chunk {selected_index}/{len(chunks)} "
                f"(original #{chunk['index']}, "
                f"({chunk['start']:.1f}s-{chunk['end']:.1f}s)"
            )
            text = transcriber.transcribe_chunk(chunk["path"])
            segments.append({"start": chunk["start"], "end": chunk["end"], "text": text})

    cleaned_segments, raw_segments = prepare_transcript_segments(segments)
    return {
        "segments": cleaned_segments,
        "raw_segments": raw_segments,
        "text": _segment_text(cleaned_segments),
        "language": None,
        "metadata": {
            "backend": "gemma_audio_transformers",
            "model_id": model_id,
            "chunk_seconds": chunk_seconds,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "torch_dtype": torch_dtype,
            "start_chunk": start_chunk,
            "max_chunks": max_chunks,
            "processed_audio_seconds": round(selected_duration, 3),
            "available_chunks": len(all_chunks),
            "selected_chunks": selected_chunk_meta,
            "local_files_only": local_files_only,
            "word_timestamps_available": False,
        },
    }


def _timed(name: str, fn, *args, **kwargs) -> TimedResult:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    seconds = time.perf_counter() - start
    return TimedResult(name=name, seconds=seconds, result=result)


def _worker_payload(name: str, seconds: float, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "elapsed_seconds": round(seconds, 3),
        "result": result,
    }


def _write_worker_output(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_worker_output(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _processed_audio_seconds(result: Dict[str, Any], fallback: float) -> float:
    metadata = result.get("metadata") if isinstance(result, dict) else None
    if isinstance(metadata, dict):
        value = metadata.get("processed_audio_seconds")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return fallback


def _offset_result_timestamps(result: Dict[str, Any], offset_seconds: float) -> Dict[str, Any]:
    if not offset_seconds:
        return result

    shifted = dict(result)
    shifted["segments"] = _offset_segments(result.get("segments", []), offset_seconds)
    shifted["raw_segments"] = _offset_segments(result.get("raw_segments", []), offset_seconds)
    shifted["text"] = _segment_text(shifted["segments"])
    metadata = dict(result.get("metadata") or {})
    metadata["timestamp_offset_seconds"] = offset_seconds
    shifted["metadata"] = metadata
    return shifted


def _run_worker_mode(args: argparse.Namespace) -> int:
    if not args.worker_output:
        raise ValueError("--worker-output is required in worker mode.")

    if args.worker == "existing":
        timed = _timed(
            "existing",
            run_existing_asr,
            os.path.abspath(args.audio),
            backend=args.existing_backend,
            model_size=args.whisper_model,
            language=args.language,
            batch_size=args.batch_size,
        )
    elif args.worker == "gemma":
        timed = _timed(
            "gemma",
            run_gemma_asr,
            os.path.abspath(args.audio),
            model_id=args.gemma_model,
            chunk_seconds=args.gemma_chunk_seconds,
            prompt=args.gemma_prompt,
            max_new_tokens=args.gemma_max_new_tokens,
            device=args.gemma_device,
            torch_dtype=args.gemma_torch_dtype,
            chunk_root=args.chunk_root,
            start_chunk=args.gemma_start_chunk,
            max_chunks=args.gemma_max_chunks,
            list_chunks=args.gemma_list_chunks,
            local_files_only=args.gemma_local_files_only,
        )
    else:
        raise ValueError(f"Unknown worker mode: {args.worker}")

    _write_worker_output(args.worker_output, _worker_payload(timed.name, timed.seconds, timed.result))
    return 0


def _run_pipeline_subprocess(kind: str, args: argparse.Namespace, clip_audio_path: str, output_dir: str) -> Dict[str, Any]:
    worker_output = os.path.join(output_dir, ".cache", f"{kind}_worker_result.json")
    Path(worker_output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "diarize_gui.compare_transcription_pipelines",
        clip_audio_path,
        "--worker",
        kind,
        "--worker-output",
        worker_output,
    ]

    if kind == "existing":
        cmd.extend(["--existing-backend", args.existing_backend, "--whisper-model", args.whisper_model])
        if args.language:
            cmd.extend(["--language", args.language])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
    elif kind == "gemma":
        cmd.extend(
            [
                "--gemma-model",
                args.gemma_model,
                "--gemma-chunk-seconds",
                str(args.gemma_chunk_seconds),
                "--gemma-prompt",
                args.gemma_prompt,
                "--gemma-max-new-tokens",
                str(args.gemma_max_new_tokens),
                "--gemma-device",
                args.gemma_device,
                "--gemma-torch-dtype",
                args.gemma_torch_dtype,
                "--chunk-root",
                os.path.join(output_dir, ".cache", "gemma_chunks"),
                "--gemma-start-chunk",
                str(args.gemma_start_chunk),
            ]
        )
        if args.gemma_max_chunks is not None:
            cmd.extend(["--gemma-max-chunks", str(args.gemma_max_chunks)])
        if args.gemma_list_chunks:
            cmd.append("--gemma-list-chunks")
        if args.gemma_local_files_only:
            cmd.append("--gemma-local-files-only")
    else:
        raise ValueError(f"Unknown pipeline kind: {kind}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        if exc.returncode < 0:
            raise RuntimeError(
                f"{kind} worker exited after signal {-exc.returncode}. "
                "This is often a native library crash; try --in-process only for debugging, "
                "or run the other backend with --skip-existing/--skip-gemma to isolate it."
            ) from exc
        raise
    return _read_worker_output(worker_output)


def _write_outputs(output_dir: str, payload: Dict[str, Any]) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "transcription_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for key in ("existing", "gemma"):
        if key not in payload:
            continue
        result = payload.get(key, {}).get("result", {})
        with open(os.path.join(output_dir, f"{key}_transcript.txt"), "w", encoding="utf-8") as f:
            f.write((result.get("text") or "").strip() + "\n")
        with open(os.path.join(output_dir, f"{key}_segments.json"), "w", encoding="utf-8") as f:
            json.dump(result.get("segments", []), f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare this project's existing ASR path with chunked Gemma audio transcription."
    )
    parser.add_argument("audio", help="Lesson audio file to transcribe.")
    parser.add_argument("--output-dir", default=None, help="Directory for comparison outputs.")
    parser.add_argument("--existing-backend", default="auto", choices=["auto", "mlx", "whisper_mps", "whisperx"])
    parser.add_argument("--whisper-model", default="small", help="Existing ASR model size/ref, e.g. small or large-v3.")
    parser.add_argument("--language", default=None, help="Optional language code passed to the existing ASR backend.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size for WhisperX/MLX paths.")
    parser.add_argument("--clip-start-seconds", type=float, default=0.0, help="Start comparing at this offset in the source audio.")
    parser.add_argument("--clip-max-seconds", type=float, default=None, help="Only compare this many seconds of source audio.")
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL, help="Audio-capable Gemma HF model id or local path.")
    parser.add_argument("--gemma-chunk-seconds", type=float, default=30.0, help="Chunk length, max 30 seconds.")
    parser.add_argument("--gemma-prompt", default=DEFAULT_GEMMA_PROMPT)
    parser.add_argument("--gemma-max-new-tokens", type=int, default=256)
    parser.add_argument("--gemma-device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--gemma-torch-dtype", default="auto", help="auto, float16, bfloat16, or float32.")
    parser.add_argument("--gemma-start-chunk", type=int, default=0, help="0-based Gemma chunk index to start from.")
    parser.add_argument("--gemma-max-chunks", type=int, default=None, help="Only transcribe this many Gemma chunks.")
    parser.add_argument("--gemma-list-chunks", action="store_true", help="Write selected Gemma chunk metadata without loading the model.")
    parser.add_argument("--gemma-local-files-only", action="store_true", help="Use only cached Gemma files; fail instead of downloading model weights.")
    parser.add_argument("--skip-existing", action="store_true", help="Only run the Gemma transcription path.")
    parser.add_argument("--skip-gemma", action="store_true", help="Only run the existing ASR path.")
    parser.add_argument("--in-process", action="store_true", help="Run both paths in this Python process. Mostly useful for debugging.")
    parser.add_argument("--worker", choices=["existing", "gemma"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-root", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audio_path = os.path.abspath(args.audio)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if args.worker:
        return _run_worker_mode(args)
    if args.skip_existing and args.skip_gemma:
        raise ValueError("At least one transcription path must run.")

    output_dir = args.output_dir or os.path.join(
        os.getcwd(),
        "transcription_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    cache_dir = os.path.join(output_dir, ".cache")

    _status("normalizing audio to mono 16 kHz")
    preprocess = preprocess_audio_mono_16k(audio_path, cache_dir)
    normalized_info = sf.info(preprocess.normalized_path)
    audio_duration = float(normalized_info.frames) / float(normalized_info.samplerate)
    clip = _clip_audio(
        preprocess.normalized_path,
        start_seconds=args.clip_start_seconds,
        max_seconds=args.clip_max_seconds,
        output_dir=cache_dir,
    )
    comparison_audio_path = clip["path"]
    comparison_duration = float(clip["duration_seconds"])

    payload: Dict[str, Any] = {
        "audio": {
            "source_path": audio_path,
            "normalized_path": preprocess.normalized_path,
            "duration_seconds": round(audio_duration, 3),
            "sample_rate": int(normalized_info.samplerate),
            "channels": int(normalized_info.channels),
            "comparison_path": comparison_audio_path,
            "comparison_start_seconds": round(float(clip["start_seconds"]), 3),
            "comparison_duration_seconds": round(comparison_duration, 3),
            "comparison_end_seconds": round(float(clip["end_seconds"]), 3),
            "clipped": bool(clip["clipped"]),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if not args.skip_existing:
        _status("running existing ASR path")
        if args.in_process:
            existing_timed = _timed(
                "existing",
                run_existing_asr,
                comparison_audio_path,
                backend=args.existing_backend,
                model_size=args.whisper_model,
                language=args.language,
                batch_size=args.batch_size,
            )
            existing = _worker_payload(existing_timed.name, existing_timed.seconds, existing_timed.result)
        else:
            existing = _run_pipeline_subprocess("existing", args, comparison_audio_path, output_dir)
        existing_result = _offset_result_timestamps(existing["result"], float(clip["start_seconds"]))
        payload["existing"] = {
            "elapsed_seconds": existing["elapsed_seconds"],
            "realtime_factor": round(existing["elapsed_seconds"] / max(comparison_duration, 0.001), 3),
            "processed_audio_seconds": round(comparison_duration, 3),
            "full_audio_realtime_factor": round(existing["elapsed_seconds"] / max(audio_duration, 0.001), 3),
            "result": existing_result,
        }

    if not args.skip_gemma:
        _status("running Gemma ASR path")
        if args.in_process:
            gemma_timed = _timed(
                "gemma",
                run_gemma_asr,
                comparison_audio_path,
                model_id=args.gemma_model,
                chunk_seconds=args.gemma_chunk_seconds,
                prompt=args.gemma_prompt,
                max_new_tokens=args.gemma_max_new_tokens,
                device=args.gemma_device,
                torch_dtype=args.gemma_torch_dtype,
                chunk_root=args.chunk_root,
                start_chunk=args.gemma_start_chunk,
                max_chunks=args.gemma_max_chunks,
                list_chunks=args.gemma_list_chunks,
                local_files_only=args.gemma_local_files_only,
            )
            gemma = _worker_payload(gemma_timed.name, gemma_timed.seconds, gemma_timed.result)
        else:
            gemma = _run_pipeline_subprocess("gemma", args, comparison_audio_path, output_dir)
        gemma_result = _offset_result_timestamps(gemma["result"], float(clip["start_seconds"]))
        gemma_processed_seconds = _processed_audio_seconds(gemma_result, comparison_duration)
        payload["gemma"] = {
            "elapsed_seconds": gemma["elapsed_seconds"],
            "realtime_factor": round(gemma["elapsed_seconds"] / max(gemma_processed_seconds, 0.001), 3),
            "processed_audio_seconds": round(gemma_processed_seconds, 3),
            "full_audio_realtime_factor": round(gemma["elapsed_seconds"] / max(audio_duration, 0.001), 3),
            "result": gemma_result,
        }

    if "existing" in payload and "gemma" in payload:
        payload["comparison"] = compare_texts(
            payload["existing"]["result"].get("text", ""),
            payload["gemma"]["result"].get("text", ""),
        )

    _write_outputs(output_dir, payload)
    _status(f"wrote comparison outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
