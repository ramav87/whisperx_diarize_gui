from __future__ import annotations

from typing import Any, Iterable


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_adjusted_grammar_score(
    raw: float | None,
    practice_hours_7d: float | None = None,
    topic_difficulty: float | None = None,
    fatigue: float | None = None,
) -> float | None:
    """Heuristic estimate of grammar ability after accounting for adverse context."""
    raw_value = _number_or_none(raw)
    if raw_value is None:
        return None

    practice = _number_or_none(practice_hours_7d)
    difficulty = _number_or_none(topic_difficulty)
    fatigue_value = _number_or_none(fatigue)

    practice_penalty = max(0, 3 - practice) * 1.5 if practice is not None else 0
    difficulty_penalty = max(0, difficulty - 3) * 2.0 if difficulty is not None else 0
    fatigue_penalty = fatigue_value * 1.0 if fatigue_value is not None else 0

    return clamp(raw_value + practice_penalty + difficulty_penalty + fatigue_penalty, 0, 100)


def compute_conceptual_load_score(
    topic_difficulty: float | None = None,
    idea_density: float | None = None,
    abstraction_level: float | None = None,
    cognitive_branching: float | None = None,
    technical_density: float | None = None,
    discourse_depth: float | None = None,
    lexical_retrieval_pressure: float | None = None,
) -> float | None:
    """Estimate total conceptual load on a 1-10 scale."""
    values: list[float] = []

    # Existing topic/idea fields are 1-5, so map them onto the new 1-10 scale.
    for value in (_number_or_none(topic_difficulty), _number_or_none(idea_density)):
        if value is not None:
            values.append(clamp(value, 1, 5) * 2)

    for value in (
        _number_or_none(abstraction_level),
        _number_or_none(cognitive_branching),
        _number_or_none(technical_density),
        _number_or_none(discourse_depth),
        _number_or_none(lexical_retrieval_pressure),
    ):
        if value is not None:
            values.append(clamp(value, 1, 10))

    if not values:
        return None
    return sum(values) / len(values)


def compute_advanced_adjusted_grammar_score(
    raw: float | None,
    practice_hours_7d: float | None = None,
    conceptual_load: float | None = None,
    fatigue: float | None = None,
) -> float | None:
    raw_value = _number_or_none(raw)
    if raw_value is None:
        return None

    practice = _number_or_none(practice_hours_7d)
    load = _number_or_none(conceptual_load)
    fatigue_value = _number_or_none(fatigue)

    practice_penalty = max(0, 3 - practice) * 1.5 if practice is not None else 0
    load_penalty = max(0, load - 6) * 2.0 if load is not None else 0
    fatigue_penalty = fatigue_value * 1.0 if fatigue_value is not None else 0
    return clamp(raw_value + practice_penalty + load_penalty + fatigue_penalty, 0, 100)


def compute_difficulty_adjusted_wpm(
    raw_wpm: float | None,
    topic_difficulty: float | None = None,
) -> float | None:
    raw_value = _number_or_none(raw_wpm)
    if raw_value is None:
        return None
    difficulty = _number_or_none(topic_difficulty)
    if difficulty is None:
        return None
    difficulty_bonus = max(0, difficulty - 3) * 8
    return raw_value + difficulty_bonus


def compute_cognitive_load_adjusted_wpm(
    raw_wpm: float | None,
    topic_difficulty: float | None = None,
    idea_density: float | None = None,
    conceptual_load: float | None = None,
) -> float | None:
    raw_value = _number_or_none(raw_wpm)
    if raw_value is None:
        return None

    load = _number_or_none(conceptual_load)
    if load is not None:
        return raw_value + max(0, load - 6) * 6

    available = [
        value
        for value in (_number_or_none(topic_difficulty), _number_or_none(idea_density))
        if value is not None
    ]
    if not available:
        return None

    load_1_to_5 = sum(available) / len(available)
    return raw_value + max(0, load_1_to_5 - 3) * 8


def compute_fluency_under_load(
    raw_wpm: float | None,
    topic_difficulty: float | None = None,
    idea_density: float | None = None,
    long_pauses_per_min: float | None = None,
    conceptual_load: float | None = None,
) -> float | None:
    raw_value = _number_or_none(raw_wpm)
    if raw_value is None:
        return None

    available = [
        value
        for value in (_number_or_none(topic_difficulty), _number_or_none(idea_density))
        if value is not None
    ]
    pause_value = _number_or_none(long_pauses_per_min)
    if not available and pause_value is None:
        return None

    load = _number_or_none(conceptual_load)
    if load is not None:
        adjusted = raw_value + max(0, load - 6) * 6
    else:
        load_1_to_5 = sum(available) / len(available) if available else 3
        adjusted = raw_value + max(0, load_1_to_5 - 3) * 8
    pause_penalty = (pause_value or 0) * 3
    return max(0, adjusted - pause_penalty)


def compute_effective_fluency_score(
    raw_wpm: float | None,
    raw_grammar_score: float | None = None,
    conceptual_load: float | None = None,
    self_repairs_per_min: float | None = None,
    long_pauses_per_min: float | None = None,
    filled_pauses_per_min: float | None = None,
) -> float | None:
    """Composite 0-100 estimate of fluent idea delivery under conceptual load."""
    raw_value = _number_or_none(raw_wpm)
    grammar = _number_or_none(raw_grammar_score)
    load = _number_or_none(conceptual_load)
    if raw_value is None or grammar is None or load is None:
        return None

    wpm_component = clamp(raw_value, 40, 120) / 120 * 35
    grammar_component = clamp(grammar, 0, 100) * 0.35
    load_component = clamp(load, 1, 10) * 3.5
    repair_penalty = (
        (_number_or_none(self_repairs_per_min) or 0) * 2.5
        + (_number_or_none(long_pauses_per_min) or 0) * 3.0
        + (_number_or_none(filled_pauses_per_min) or 0) * 1.0
    )
    return clamp(wpm_component + grammar_component + load_component - repair_penalty, 0, 100)


def compute_complexity_resilience_score(
    raw_grammar_score: float | None = None,
    raw_wpm: float | None = None,
    conceptual_load: float | None = None,
) -> float | None:
    """Rewards stable grammar and speed when conceptual load is high."""
    grammar = _number_or_none(raw_grammar_score)
    wpm = _number_or_none(raw_wpm)
    load = _number_or_none(conceptual_load)
    if grammar is None or wpm is None or load is None:
        return None
    return clamp(grammar * 0.45 + clamp(wpm, 40, 120) / 120 * 30 + clamp(load, 1, 10) * 2.5, 0, 100)


def classify_context(
    practice_hours_last_7_days: float | None = None,
    topic_difficulty: float | None = None,
    idea_density: float | None = None,
) -> dict[str, bool]:
    practice = _number_or_none(practice_hours_last_7_days)
    difficulty = _number_or_none(topic_difficulty)
    density = _number_or_none(idea_density)
    return {
        "cold_session": practice is not None and practice < 1.0,
        "hard_topic": difficulty is not None and difficulty >= 4,
        "high_idea_density": density is not None and density >= 4,
    }


def build_context_metrics(
    raw_grammar_score: float | None = None,
    raw_wpm: float | None = None,
    practice_hours_last_7_days: float | None = None,
    topic_difficulty: float | None = None,
    idea_density: float | None = None,
    abstraction_level: float | None = None,
    cognitive_branching: float | None = None,
    technical_density: float | None = None,
    discourse_depth: float | None = None,
    lexical_retrieval_pressure: float | None = None,
    fatigue_or_stress: float | None = None,
    long_pauses_per_min: float | None = None,
    self_repairs_per_min: float | None = None,
    filled_pauses_per_min: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build backward-compatible heuristic metrics; missing inputs skip only dependents."""
    metrics: dict[str, Any] = {
        "raw_grammar_score": _number_or_none(raw_grammar_score),
        "raw_wpm": _number_or_none(raw_wpm),
        "practice_hours_last_7_days": _number_or_none(practice_hours_last_7_days),
        "topic_difficulty": _number_or_none(topic_difficulty),
        "idea_density": _number_or_none(idea_density),
        "abstraction_level": _number_or_none(abstraction_level),
        "cognitive_branching": _number_or_none(cognitive_branching),
        "technical_density": _number_or_none(technical_density),
        "discourse_depth": _number_or_none(discourse_depth),
        "lexical_retrieval_pressure": _number_or_none(lexical_retrieval_pressure),
        "fatigue_or_stress": _number_or_none(fatigue_or_stress),
        "long_pauses_per_min": _number_or_none(long_pauses_per_min),
        "self_repairs_per_min": _number_or_none(self_repairs_per_min),
        "filled_pauses_per_min": _number_or_none(filled_pauses_per_min),
        "notes": notes,
    }
    metrics.update(
        classify_context(
            metrics["practice_hours_last_7_days"],
            metrics["topic_difficulty"],
            metrics["idea_density"],
        )
    )

    conceptual_load = compute_conceptual_load_score(
        metrics["topic_difficulty"],
        metrics["idea_density"],
        metrics["abstraction_level"],
        metrics["cognitive_branching"],
        metrics["technical_density"],
        metrics["discourse_depth"],
        metrics["lexical_retrieval_pressure"],
    )
    if conceptual_load is not None:
        metrics["conceptual_load_score"] = conceptual_load

    has_advanced_load = any(
        metrics.get(key) is not None
        for key in (
            "abstraction_level",
            "cognitive_branching",
            "technical_density",
            "discourse_depth",
            "lexical_retrieval_pressure",
        )
    )
    if has_advanced_load:
        adjusted_grammar = compute_advanced_adjusted_grammar_score(
            metrics["raw_grammar_score"],
            metrics["practice_hours_last_7_days"],
            metrics.get("conceptual_load_score"),
            metrics["fatigue_or_stress"],
        )
    else:
        adjusted_grammar = compute_adjusted_grammar_score(
            metrics["raw_grammar_score"],
            metrics["practice_hours_last_7_days"],
            metrics["topic_difficulty"],
            metrics["fatigue_or_stress"],
        )
    if adjusted_grammar is not None:
        metrics["adjusted_grammar_score"] = adjusted_grammar

    difficulty_wpm = compute_difficulty_adjusted_wpm(metrics["raw_wpm"], metrics["topic_difficulty"])
    if difficulty_wpm is not None:
        metrics["difficulty_adjusted_wpm"] = difficulty_wpm

    cognitive_wpm = compute_cognitive_load_adjusted_wpm(
        metrics["raw_wpm"],
        metrics["topic_difficulty"],
        metrics["idea_density"],
        metrics.get("conceptual_load_score") if has_advanced_load else None,
    )
    if cognitive_wpm is not None:
        metrics["cognitive_load_adjusted_wpm"] = cognitive_wpm

    fluency = compute_fluency_under_load(
        metrics["raw_wpm"],
        metrics["topic_difficulty"],
        metrics["idea_density"],
        metrics["long_pauses_per_min"],
        metrics.get("conceptual_load_score") if has_advanced_load else None,
    )
    if fluency is not None:
        metrics["fluency_under_load"] = fluency

    effective_fluency = compute_effective_fluency_score(
        metrics["raw_wpm"],
        metrics["raw_grammar_score"],
        metrics.get("conceptual_load_score"),
        metrics["self_repairs_per_min"],
        metrics["long_pauses_per_min"],
        metrics["filled_pauses_per_min"],
    )
    if effective_fluency is not None:
        metrics["effective_fluency_score"] = effective_fluency

    resilience = compute_complexity_resilience_score(
        metrics["raw_grammar_score"],
        metrics["raw_wpm"],
        metrics.get("conceptual_load_score"),
    )
    if resilience is not None:
        metrics["complexity_resilience_score"] = resilience

    return metrics


def _score_from_session(session: dict[str, Any], use_adjusted: bool) -> float | None:
    context = session.get("context_metrics") if isinstance(session.get("context_metrics"), dict) else {}
    if use_adjusted:
        adjusted = _number_or_none(context.get("adjusted_grammar_score"))
        if adjusted is not None:
            return adjusted
    return _number_or_none(
        session.get("raw_grammar_score", session.get("grammar_score", context.get("raw_grammar_score")))
    )


def compute_automaticity_gap(
    sessions: Iterable[dict[str, Any]],
    window: int = 10,
    use_adjusted: bool = False,
) -> dict[str, Any]:
    recent = list(sessions)[-window:]
    warm_scores: list[float] = []
    cold_scores: list[float] = []

    for session in recent:
        context = session.get("context_metrics") if isinstance(session.get("context_metrics"), dict) else {}
        practice = _number_or_none(
            session.get("practice_hours_last_7_days", context.get("practice_hours_last_7_days"))
        )
        score = _score_from_session(session, use_adjusted=use_adjusted)
        if practice is None or score is None:
            continue
        if practice >= 1.0:
            warm_scores.append(score)
        else:
            cold_scores.append(score)

    warm_avg = sum(warm_scores) / len(warm_scores) if warm_scores else None
    cold_avg = sum(cold_scores) / len(cold_scores) if cold_scores else None
    gap = warm_avg - cold_avg if warm_avg is not None and cold_avg is not None else None
    return {
        "warm_adjusted_or_raw_grammar_avg": warm_avg,
        "cold_adjusted_or_raw_grammar_avg": cold_avg,
        "automaticity_gap": gap,
        "warm_session_count": len(warm_scores),
        "cold_session_count": len(cold_scores),
        "window": window,
    }


def interpretation_for_context_metrics(
    metrics: dict[str, Any] | None,
    previous_metrics: dict[str, Any] | None = None,
) -> str:
    if not metrics:
        return "Context-adjusted metrics are not available for this lesson yet."

    messages: list[str] = []
    raw_grammar = _number_or_none(metrics.get("raw_grammar_score"))
    adjusted_grammar = _number_or_none(metrics.get("adjusted_grammar_score"))
    raw_wpm = _number_or_none(metrics.get("raw_wpm"))
    cognitive_wpm = _number_or_none(metrics.get("cognitive_load_adjusted_wpm"))
    long_pauses = _number_or_none(metrics.get("long_pauses_per_min"))

    if previous_metrics:
        prev_raw_grammar = _number_or_none(previous_metrics.get("raw_grammar_score"))
        prev_adjusted = _number_or_none(previous_metrics.get("adjusted_grammar_score"))
        prev_raw_wpm = _number_or_none(previous_metrics.get("raw_wpm"))
        if (
            prev_raw_grammar is not None
            and raw_grammar is not None
            and raw_grammar < prev_raw_grammar
            and prev_adjusted is not None
            and adjusted_grammar is not None
            and adjusted_grammar >= prev_adjusted - 1
        ):
            messages.append(
                "Raw grammar dipped, but this was a cold and/or hard-topic session. "
                "Context-adjusted grammar remains consistent with your recent level."
            )
        if (
            prev_raw_wpm is not None
            and raw_wpm is not None
            and abs(raw_wpm - prev_raw_wpm) <= 3
            and cognitive_wpm is not None
            and cognitive_wpm > raw_wpm
        ):
            messages.append(
                "Raw WPM was stable while topic load increased, suggesting stable or improving fluency under load."
            )

    if raw_wpm is not None and cognitive_wpm is not None and cognitive_wpm < raw_wpm and long_pauses:
        messages.append("Speech flow may have been strained in this session, especially if the topic load was high.")
    elif long_pauses is not None and long_pauses >= 3:
        messages.append("Speech flow may have been strained in this session, especially if the topic load was high.")

    if metrics.get("cold_session"):
        messages.append("This was a cold session: low recent Spanish output may have reduced automaticity.")
    if metrics.get("hard_topic"):
        messages.append("This was a hard-topic session: abstract or technical content likely increased grammatical load.")

    if not messages:
        messages.append("Raw and context-adjusted metrics look broadly aligned for this lesson.")
    return " ".join(messages)
