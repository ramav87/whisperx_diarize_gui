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

import base64

_OBFUSCATION_PREFIX = "obf:v1:"

def obfuscate_secret(secret: str) -> str:
    """
    Lightweight obfuscation (NOT encryption).
    Prevents casual reading of secrets in config files.
    """
    if not secret:
        return ""
    raw = secret.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return _OBFUSCATION_PREFIX + encoded


def deobfuscate_secret(value: str) -> str:
    if not value:
        return ""
    if not value.startswith(_OBFUSCATION_PREFIX):
        # Backward compatibility / plain text
        return value
    encoded = value[len(_OBFUSCATION_PREFIX):]
    try:
        return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
    except Exception:
        return ""

from typing import Dict, Optional

# ==============================
# OpenAI pricing (USD per 1M tokens)
# ==============================

OPENAI_PRICING_PER_1M = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-5.1": {
        "input": 1.25,
        "output": 10.00,
    },
    "gpt-5.2": {
        "input": 1.75,
        "output": 14.00,
    },
}

# ==============================
# Token estimation utilities
# ==============================

def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Rough token estimate.
    Rule of thumb: ~4 characters per token for English / Spanish.
    """
    if char_count <= 0:
        return 0
    return max(1, char_count // 4)


def estimate_openai_cost(
    *,
    text: str,
    model: str,
    output_ratio: float = 0.35,
    pricing: Dict[str, Dict[str, float]] = OPENAI_PRICING_PER_1M,
) -> Optional[dict]:
    """
    Estimate OpenAI API cost for a single analysis run.

    Parameters
    ----------
    text : str
        Full input text sent to the model (prompt + transcript).
    model : str
        OpenAI model name (must exist in pricing table).
    output_ratio : float
        Estimated output tokens as a fraction of input tokens.
        0.3–0.4 is typical for analysis tasks.
    pricing : dict
        Pricing table (USD per 1M tokens).

    Returns
    -------
    dict or None
        {
            "chars": int,
            "input_tokens": int,
            "output_tokens": int,
            "input_cost_usd": float,
            "output_cost_usd": float,
            "total_cost_usd": float,
        }
        or None if model is unsupported.
    """
    if not text or model not in pricing:
        return None

    chars = len(text)
    input_tokens = estimate_tokens_from_chars(chars)
    output_tokens = int(input_tokens * output_ratio)

    p = pricing[model]

    input_cost = (input_tokens / 1_000_000) * p["input"]
    output_cost = (output_tokens / 1_000_000) * p["output"]

    return {
        "chars": chars,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }
