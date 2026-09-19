"""Deterministic, transcript-derived measures for language-learning progress."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:['’-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?")

# High-frequency Spanish function words are removed only from vocabulary views;
# they remain part of length, fluency, and diversity calculations.
SPANISH_STOPWORDS = {
    "a", "al", "algo", "como", "con", "de", "del", "desde", "donde", "el",
    "ella", "ellos", "en", "era", "es", "esa", "ese", "eso", "esta", "este",
    "esto", "fue", "ha", "hay", "la", "las", "le", "les", "lo", "los", "más",
    "me", "mi", "muy", "no", "nos", "o", "para", "pero", "por", "porque", "que",
    "qué", "se", "si", "sí", "sin", "son", "su", "sus", "también", "te", "tiene",
    "todo", "tu", "un", "una", "uno", "unos", "y", "ya", "yo",
}

CONNECTORS = {
    "además", "aunque", "entonces", "finalmente", "incluso", "mientras", "primero",
    "quizás", "también", "tampoco", "después", "luego", "sin embargo", "por eso",
    "por ejemplo", "en cambio", "así que", "a pesar", "es decir", "de hecho",
}

SUBORDINATORS = {
    "aunque", "cuando", "mientras", "porque", "si", "como", "donde", "quien",
    "quienes", "cuyo", "cuya", "cuyos", "cuyas", "para que", "antes de que",
    "después de que", "a menos que", "a pesar de que", "sin que",
}

FILLERS = {"eh", "em", "este", "esto", "mmm", "mm", "pues", "bueno"}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def moving_average_type_token_ratio(tokens: list[str], window: int = 50) -> float | None:
    """Return MATTR; short samples use their transparent whole-sample TTR."""
    if not tokens:
        return None
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [
        len(set(tokens[index:index + window])) / window
        for index in range(len(tokens) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def _phrase_count(text: str, phrases: Iterable[str]) -> tuple[int, set[str]]:
    normalized = re.sub(r"\s+", " ", text.lower())
    lowered = f" {normalized} "
    used: set[str] = set()
    count = 0
    for phrase in phrases:
        matches = len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered))
        if matches:
            used.add(phrase)
            count += matches
    return count, used


def learner_segments(
    segments: Iterable[dict[str, Any]],
    student_speakers: Iterable[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Select learner turns and expose whether speaker identity was explicit or inferred."""
    student_ids = {str(value) for value in (student_speakers or []) if str(value).strip()}
    valid_segments = [segment for segment in segments if isinstance(segment, dict)]
    if student_ids:
        return (
            [segment for segment in valid_segments if str(segment.get("speaker")) in student_ids],
            "explicit",
        )

    inferred_ids = {
        str(segment.get("speaker"))
        for segment in valid_segments
        if "01" in str(segment.get("speaker", ""))
    }
    if inferred_ids:
        return (
            [segment for segment in valid_segments if str(segment.get("speaker")) in inferred_ids],
            "inferred_speaker_01",
        )
    return [], "unresolved"


def compute_language_growth_metrics(
    segments: Iterable[dict[str, Any]],
    student_speakers: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute descriptive learner-language metrics without calling an LLM."""
    selected, scope = learner_segments(segments, student_speakers)
    utterances = [str(segment.get("text", "")).strip() for segment in selected]
    utterances = [text for text in utterances if text]
    tokens_by_utterance = [tokenize(text) for text in utterances]
    tokens = [token for utterance in tokens_by_utterance for token in utterance]
    token_count = len(tokens)

    content_tokens = [
        token for token in tokens
        if token not in SPANISH_STOPWORDS and token not in FILLERS and len(token) > 2
    ]
    connector_count, connectors_used = _phrase_count(" ".join(utterances), CONNECTORS)
    subordinator_count, subordinators_used = _phrase_count(" ".join(utterances), SUBORDINATORS)
    filler_count = sum(1 for token in tokens if token in FILLERS)
    lengths = [len(item) for item in tokens_by_utterance if item]

    per_100 = lambda count: (count / token_count * 100.0) if token_count else None
    return {
        "schema_version": 1,
        "speaker_scope": scope,
        "utterance_count": len(lengths),
        "token_count": token_count,
        "unique_word_forms": len(set(tokens)),
        "mattr_50": moving_average_type_token_ratio(tokens, 50),
        "mean_utterance_words": (sum(lengths) / len(lengths)) if lengths else None,
        "long_utterance_pct": (
            sum(length >= 15 for length in lengths) / len(lengths) * 100.0
            if lengths else None
        ),
        "connector_types": len(connectors_used),
        "connectors_per_100_words": per_100(connector_count),
        "subordinator_types": len(subordinators_used),
        "subordinators_per_100_words": per_100(subordinator_count),
        "filled_pauses_per_100_words": per_100(filler_count),
        "long_words_per_100_words": per_100(sum(len(token) >= 8 for token in content_tokens)),
        "top_content_words": [
            {"word": word, "count": count}
            for word, count in Counter(content_tokens).most_common(40)
        ],
        "limitations": [
            "Utterances follow ASR/diarization segments, not parser-derived sentences.",
            "Word forms are lowercased but not lemmatized.",
            *( ["Learner speaker was inferred as a speaker ID containing 01."] if scope == "inferred_speaker_01" else [] ),
            *( ["Learner speaker could not be identified."] if scope == "unresolved" else [] ),
        ],
    }
