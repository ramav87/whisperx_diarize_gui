from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import soundfile as sf


@dataclass
class AudioPreprocessResult:
    source_path: str
    normalized_path: str
    reused_cache: bool
    already_normalized: bool
    sample_rate: int
    channels: int


def _safe_stat(path: str):
    try:
        return os.stat(path)
    except Exception:
        return None


def _cache_key(source_path: str) -> str:
    stat = _safe_stat(source_path)
    payload = {
        "path": os.path.abspath(source_path),
        "mtime_ns": getattr(stat, "st_mtime_ns", None),
        "size": getattr(stat, "st_size", None),
        "target_sr": 16000,
        "target_channels": 1,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def preprocess_audio_mono_16k(
    source_path: str,
    cache_dir: str,
    *,
    force: bool = False,
) -> AudioPreprocessResult:
    """
    Normalize audio to mono 16 kHz WAV and cache the result.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    os.makedirs(cache_dir, exist_ok=True)

    try:
        info = sf.info(source_path)
        if (
            not force
            and info.format == "WAV"
            and info.channels == 1
            and int(info.samplerate) == 16000
        ):
            return AudioPreprocessResult(
                source_path=source_path,
                normalized_path=source_path,
                reused_cache=False,
                already_normalized=True,
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
            )
    except Exception:
        info = None

    cache_name = f"{_cache_key(source_path)}_mono16k.wav"
    normalized_path = os.path.join(cache_dir, cache_name)

    if os.path.isfile(normalized_path) and not force:
        try:
            cached_info = sf.info(normalized_path)
            if cached_info.channels == 1 and int(cached_info.samplerate) == 16000:
                return AudioPreprocessResult(
                    source_path=source_path,
                    normalized_path=normalized_path,
                    reused_cache=True,
                    already_normalized=False,
                    sample_rate=int(cached_info.samplerate),
                    channels=int(cached_info.channels),
                )
        except Exception:
            pass

    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required to normalize audio for transcription and diarization.")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        normalized_path,
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cached_info = sf.info(normalized_path)
    return AudioPreprocessResult(
        source_path=source_path,
        normalized_path=normalized_path,
        reused_cache=False,
        already_normalized=False,
        sample_rate=int(cached_info.samplerate),
        channels=int(cached_info.channels),
    )
