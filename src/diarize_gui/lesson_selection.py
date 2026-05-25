from __future__ import annotations

import json
import os


def ai_stats_complete(ai_path: str) -> bool:
    if not os.path.isfile(ai_path):
        return False

    try:
        with open(ai_path, "r", encoding="utf-8") as f:
            stats = json.load(f) or {}
    except Exception:
        return False

    if not isinstance(stats.get("grammar_score"), (int, float)):
        return False

    context = stats.get("context_metrics") if isinstance(stats.get("context_metrics"), dict) else {}
    required_context_keys = (
        "raw_grammar_score",
        "adjusted_grammar_score",
        "topic_difficulty",
        "idea_density",
        "abstraction_level",
        "cognitive_branching",
        "technical_density",
        "discourse_depth",
        "lexical_retrieval_pressure",
        "conceptual_load_score",
    )
    return all(isinstance(context.get(key), (int, float)) for key in required_context_keys)


def select_pending_ai_lesson_dirs(lessons_dir: str) -> list[str]:
    """
    Return the newest contiguous lessons that still need complete AI metrics.

    We walk newest-first and stop once we reach the first lesson that already
    has complete ai_stats.json. That keeps batch recomputation focused on the
    newest incomplete backlog instead of reprocessing the entire history.
    """
    if not os.path.isdir(lessons_dir):
        return []

    pending: list[str] = []
    for lid in sorted(os.listdir(lessons_dir), reverse=True):
        path = os.path.join(lessons_dir, lid)
        if not os.path.isdir(path):
            continue

        meta_path = os.path.join(path, "meta.json")
        seg_path = os.path.join(path, "segments.json")
        ai_path = os.path.join(path, "ai_stats.json")
        if not (os.path.isfile(meta_path) and os.path.isfile(seg_path)):
            continue

        if ai_stats_complete(ai_path):
            break

        pending.append(path)

    return pending


def select_all_incomplete_ai_lesson_dirs(lessons_dir: str) -> list[str]:
    """Return every lesson that can be analyzed and lacks complete AI metrics."""
    if not os.path.isdir(lessons_dir):
        return []

    pending: list[str] = []
    for lid in sorted(os.listdir(lessons_dir), reverse=True):
        path = os.path.join(lessons_dir, lid)
        if not os.path.isdir(path):
            continue

        meta_path = os.path.join(path, "meta.json")
        seg_path = os.path.join(path, "segments.json")
        ai_path = os.path.join(path, "ai_stats.json")
        if not (os.path.isfile(meta_path) and os.path.isfile(seg_path)):
            continue

        if not ai_stats_complete(ai_path):
            pending.append(path)

    return pending
