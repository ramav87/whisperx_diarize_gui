# src/diarize_gui/openai_provider.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests


class OpenAIProviderError(RuntimeError):
    pass


@dataclass
class OpenAIProvider:
    """
    Minimal OpenAI "Responses API" client for text-only analysis.

    Uses:
      POST https://api.openai.com/v1/responses
    Docs:
      https://platform.openai.com/docs/api-reference/responses
    """
    api_key: str
    model: str = "gpt-5.2"  # you can override per-call too
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 600

    def analyze(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.api_key or not self.api_key.strip():
            raise OpenAIProviderError("OpenAI API key is missing.")

        if not prompt or not prompt.strip():
            raise OpenAIProviderError("Prompt is empty.")

        url = f"{self.base_url.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": (model or self.model),
            "input": prompt,
        }

        # Optional knobs (only include if provided)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_output_tokens is not None:
            payload["max_output_tokens"] = int(max_output_tokens)

        if extra:
            # Allow callers to pass advanced parameters without changing this class
            payload.update(extra)

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise OpenAIProviderError(f"Network error calling OpenAI: {e}") from e

        # Raise detailed errors
        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = {"raw": resp.text}
            raise OpenAIProviderError(
                f"OpenAI API error {resp.status_code}: {err}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise OpenAIProviderError(f"Could not parse OpenAI JSON response: {e}") from e

        text = self._extract_text(data)
        if not text:
            raise OpenAIProviderError(f"OpenAI response missing text. Raw: {data}")

        return text.strip()

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        """
        Robust extraction for Responses API.
        Prefer:
          - data["output_text"] (commonly present)
        Else fall back to scanning `output` blocks for text segments.
        """
        # Most convenient form (commonly available)
        if isinstance(data, dict) and isinstance(data.get("output_text"), str):
            return data["output_text"]

        # Otherwise, walk output blocks
        out = data.get("output")
        if isinstance(out, list):
            chunks = []
            for item in out:
                # item might contain: {"content":[{"type":"output_text","text":"..."}], ...}
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        # Typical: {"type":"output_text","text":"..."}
                        if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                            chunks.append(c["text"])
                        # Some shapes nest a "text" object; handle lightly
                        elif isinstance(c.get("text"), dict) and isinstance(c["text"].get("value"), str):
                            chunks.append(c["text"]["value"])
            return "\n".join(chunks)

        # Last resort: older "choices" style (if it ever shows up via proxies)
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            ch0 = data["choices"][0]
            if isinstance(ch0, dict):
                msg = ch0.get("message", {})
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
                if isinstance(ch0.get("text"), str):
                    return ch0["text"]

        return ""
