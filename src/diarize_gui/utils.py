"""
Utility functions for device selection and timestamp formatting.
"""

import torch


def detect_device() -> str:
    """
    Choose best available device for WhisperX: cuda > cpu.
    (We deliberately do NOT return 'mps' because WhisperX
    does not support it and will raise 'unsupported device mps'.)
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) to HH:MM:SS.mmm
    """
    if seconds is None:
        return "00:00:00.000"

    ms = int(round(seconds * 1000))
    s = ms // 1000
    ms = ms % 1000
    h = s // 3600
    s = s % 3600
    m = s // 60
    s = s % 60

    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
