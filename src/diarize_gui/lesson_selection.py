from __future__ import annotations

import os


def select_pending_ai_lesson_dirs(lessons_dir: str) -> list[str]:
    """
    Return the newest contiguous lessons that still need AI metrics.

    We walk newest-first and stop once we reach the first lesson that already
    has ai_stats.json. That keeps batch recomputation focused on the newest
    unprocessed backlog instead of reprocessing the entire history.
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

        if os.path.isfile(ai_path):
            break

        pending.append(path)

    return pending
