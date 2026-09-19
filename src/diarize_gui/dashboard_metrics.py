from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .metrics.context_adjusted import build_context_metrics, compute_automaticity_gap
from .metrics.language_growth import compute_language_growth_metrics


def build_profile_dashboard(profile_id: str | None, lessons_dir: Path) -> dict[str, Any]:
    safe_profile = str(profile_id or "default")
    lesson_dirs = _profile_lesson_dirs(safe_profile, lessons_dir)
    practice_hours_by_lesson = _practice_hours_by_lesson(lesson_dirs)

    total_recording_sec = 0.0
    student_speaking_sec = 0.0
    student_total_words = 0
    total_latency_sum = 0.0
    total_latency_count = 0
    max_turn_duration = 0.0
    analyzed_count = 0
    ai_v2_count = 0
    explicit_ai_scope_count = 0

    all_grammar_scores: list[tuple[datetime, float, str]] = []
    golden_words_all: list[str] = []
    context_sessions: list[dict[str, Any]] = []
    context_trend: list[dict[str, Any]] = []
    student_words_by_month: dict[str, int] = defaultdict(int)
    fluency_trend: list[dict[str, Any]] = []
    language_growth_trend: list[dict[str, Any]] = []
    vocabulary_by_month: dict[str, Counter[str]] = defaultdict(Counter)
    lessons: list[dict[str, Any]] = []

    for lesson_dir in lesson_dirs:
        meta = _read_json(lesson_dir / "meta.json") or {}
        segments = _read_json(lesson_dir / "segments.json") or []
        if not isinstance(meta, dict) or not isinstance(segments, list):
            continue

        lesson_id = lesson_dir.name
        dt_obj = _parse_lesson_datetime(lesson_id, meta)
        date_value = dt_obj.isoformat() if dt_obj else None
        month_key = dt_obj.strftime("%Y-%m") if dt_obj else "Unknown"
        ai_data = _read_json(lesson_dir / "ai_stats.json") or {}
        if not isinstance(ai_data, dict):
            ai_data = {}
        if ai_data:
            analyzed_count += 1
            if ai_data.get("analysis_schema_version") == 2:
                ai_v2_count += 1
            provenance = ai_data.get("analysis_provenance") if isinstance(ai_data.get("analysis_provenance"), dict) else {}
            scope = ai_data.get("analysis_scope") if isinstance(ai_data.get("analysis_scope"), dict) else {}
            if provenance.get("student_scope") == "explicit" or scope.get("speaker_scope") == "explicit":
                explicit_ai_scope_count += 1

        grammar_score = _number_or_none(ai_data.get("grammar_score"))
        if grammar_score is not None and dt_obj:
            all_grammar_scores.append((dt_obj, grammar_score, lesson_id))

        golden_words = ai_data.get("golden_words", [])
        if isinstance(golden_words, str):
            golden_words = [golden_words]
        if isinstance(golden_words, list):
            golden_words_all.extend(str(word).strip() for word in golden_words if str(word).strip())

        lesson_dur = _number_or_none(meta.get("duration_sec")) or 0.0
        total_recording_sec += max(0.0, lesson_dur)

        student_ids = {str(speaker) for speaker in meta.get("student_speakers", [])}
        lesson_student_words = 0
        lesson_student_sec = 0.0
        lesson_lat_sum = 0.0
        lesson_lat_cnt = 0
        lesson_max_turn = 0.0
        last_end = 0.0
        last_was_student = False

        for index, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            speaker = str(seg.get("speaker", "UNKNOWN"))
            start = _number_or_none(seg.get("start")) or 0.0
            end = _number_or_none(seg.get("end")) or 0.0
            duration = max(0.0, end - start)
            text = str(seg.get("text", "")).strip()
            is_student = (speaker in student_ids) or (not student_ids and "01" in speaker)

            if is_student:
                lesson_student_sec += duration
                word_count = len(text.split())
                lesson_student_words += word_count
                student_total_words += word_count
                lesson_max_turn = max(lesson_max_turn, duration)
                max_turn_duration = max(max_turn_duration, duration)
                if index > 0 and not last_was_student:
                    latency = start - last_end
                    if 0.0 < latency < 10.0:
                        lesson_lat_sum += latency
                        lesson_lat_cnt += 1
                last_was_student = True
            else:
                last_was_student = False
            last_end = end

        student_words_by_month[month_key] += lesson_student_words
        total_latency_sum += lesson_lat_sum
        total_latency_count += lesson_lat_cnt
        student_speaking_sec += lesson_student_sec

        lesson_raw_wpm = (lesson_student_words / (lesson_student_sec / 60.0)) if lesson_student_sec > 10 else None
        lesson_avg_latency = (lesson_lat_sum / lesson_lat_cnt) if lesson_lat_cnt else None
        if lesson_raw_wpm is not None and dt_obj:
            fluency_trend.append(
                {
                    "lesson_id": lesson_id,
                    "date": date_value,
                    "raw_wpm": lesson_raw_wpm,
                    "avg_latency_sec": lesson_avg_latency,
                }
            )

        language_metrics = compute_language_growth_metrics(
            segments,
            meta.get("student_speakers") or [],
        )
        if dt_obj and language_metrics.get("token_count"):
            language_growth_trend.append(
                {
                    "lesson_id": lesson_id,
                    "date": date_value,
                    **{
                        key: value
                        for key, value in language_metrics.items()
                        if key not in {"top_content_words", "limitations"}
                    },
                }
            )
            for item in language_metrics.get("top_content_words", []):
                if isinstance(item, dict) and item.get("word"):
                    vocabulary_by_month[month_key][str(item["word"])] += int(item.get("count") or 0)

        context_metrics = None
        if dt_obj:
            existing_context = ai_data.get("context_metrics") if isinstance(ai_data.get("context_metrics"), dict) else {}
            context_metrics = build_context_metrics(
                raw_grammar_score=ai_data.get("grammar_score", existing_context.get("raw_grammar_score")),
                raw_wpm=lesson_raw_wpm if lesson_raw_wpm is not None else existing_context.get("raw_wpm"),
                practice_hours_last_7_days=practice_hours_by_lesson.get(lesson_id),
                topic_difficulty=ai_data.get("topic_difficulty", existing_context.get("topic_difficulty")),
                idea_density=ai_data.get("idea_density", existing_context.get("idea_density")),
                abstraction_level=ai_data.get("abstraction_level", existing_context.get("abstraction_level")),
                cognitive_branching=ai_data.get("cognitive_branching", existing_context.get("cognitive_branching")),
                technical_density=ai_data.get("technical_density", existing_context.get("technical_density")),
                discourse_depth=ai_data.get("discourse_depth", existing_context.get("discourse_depth")),
                lexical_retrieval_pressure=ai_data.get(
                    "lexical_retrieval_pressure",
                    existing_context.get("lexical_retrieval_pressure"),
                ),
                fatigue_or_stress=existing_context.get("fatigue_or_stress"),
                long_pauses_per_min=existing_context.get("long_pauses_per_min"),
                self_repairs_per_min=existing_context.get("self_repairs_per_min"),
                filled_pauses_per_min=existing_context.get("filled_pauses_per_min"),
                notes=ai_data.get("context_notes", existing_context.get("notes")),
            )
            context_sessions.append(
                {
                    "date": dt_obj,
                    "grammar_score": ai_data.get("grammar_score"),
                    "context_metrics": context_metrics,
                }
            )
            context_trend.append({"lesson_id": lesson_id, "date": date_value, "context_metrics": context_metrics})

        lessons.append(
            {
                "id": lesson_id,
                "date": date_value,
                "duration_sec": lesson_dur,
                "student_speaking_sec": lesson_student_sec,
                "student_words": lesson_student_words,
                "raw_wpm": lesson_raw_wpm,
                "avg_latency_sec": lesson_avg_latency,
                "max_turn_duration_sec": lesson_max_turn,
                "grammar_score": grammar_score,
                "context_metrics": context_metrics,
                "language_metrics": language_metrics,
                "analyzed": bool(ai_data),
            }
        )

    total_hours = total_recording_sec / 3600.0
    student_speaking_pct = (student_speaking_sec / total_recording_sec * 100.0) if total_recording_sec else 0.0
    global_wpm = (student_total_words / (student_speaking_sec / 60.0)) if student_speaking_sec > 30 else 0.0
    avg_latency = (total_latency_sum / total_latency_count) if total_latency_count else 0.0
    grammar_scores = [score for _, score, _ in all_grammar_scores]
    average_grammar = sum(grammar_scores) / len(grammar_scores) if grammar_scores else None
    automaticity_gap = compute_automaticity_gap(
        sorted(context_sessions, key=lambda item: item["date"]),
        window=10,
    )

    return {
        "profile": safe_profile,
        "summary": {
            "lesson_count": len(lesson_dirs),
            "analyzed_lesson_count": analyzed_count,
            "ai_metrics_v2_count": ai_v2_count,
            "legacy_ai_metrics_count": max(0, analyzed_count - ai_v2_count),
            "explicit_ai_scope_count": explicit_ai_scope_count,
            "total_recording_sec": total_recording_sec,
            "total_hours": total_hours,
            "student_speaking_sec": student_speaking_sec,
            "student_speaking_pct": student_speaking_pct,
            "student_total_words": student_total_words,
            "global_wpm": global_wpm,
            "avg_latency_sec": avg_latency,
            "max_turn_duration_sec": max_turn_duration,
            "average_grammar_score": average_grammar,
            "automaticity_gap": automaticity_gap,
            "golden_words": _latest_unique(golden_words_all, limit=10),
        },
        "trends": {
            "activity": [
                {"month": month, "student_words": words}
                for month, words in sorted(student_words_by_month.items())
            ],
            "fluency": sorted(fluency_trend, key=lambda item: item["date"] or ""),
            "grammar": [
                {"lesson_id": lesson_id, "date": dt_obj.isoformat(), "grammar_score": score}
                for dt_obj, score, lesson_id in sorted(all_grammar_scores, key=lambda item: item[0])
            ],
            "context": sorted(context_trend, key=lambda item: item["date"] or ""),
            "language_growth": sorted(language_growth_trend, key=lambda item: item["date"] or ""),
            "vocabulary": [
                {
                    "month": month,
                    "words": [
                        {"word": word, "count": count}
                        for word, count in counts.most_common(40)
                    ],
                }
                for month, counts in sorted(vocabulary_by_month.items())
            ],
        },
        "lessons": sorted(lessons, key=lambda item: item["date"] or item["id"], reverse=True),
    }


def _profile_lesson_dirs(profile_id: str, lessons_dir: Path) -> list[Path]:
    if not lessons_dir.is_dir():
        return []
    lesson_dirs = []
    for lesson_dir in lessons_dir.iterdir():
        if not lesson_dir.is_dir():
            continue
        meta = _read_json(lesson_dir / "meta.json") or {}
        if isinstance(meta, dict) and meta.get("profile") == profile_id:
            lesson_dirs.append(lesson_dir)
    return sorted(lesson_dirs, key=lambda item: item.name)


def _practice_hours_by_lesson(lesson_dirs: list[Path]) -> dict[str, float]:
    lesson_times: list[tuple[str, datetime, float]] = []
    for lesson_dir in lesson_dirs:
        meta = _read_json(lesson_dir / "meta.json") or {}
        if not isinstance(meta, dict):
            continue
        dt_obj = _parse_lesson_datetime(lesson_dir.name, meta)
        duration_sec = _number_or_none(meta.get("duration_sec")) or 0.0
        if dt_obj:
            lesson_times.append((lesson_dir.name, dt_obj, max(0.0, duration_sec)))

    practice_hours = {}
    for lesson_id, dt_obj, _duration_sec in lesson_times:
        window_start = dt_obj - timedelta(days=7)
        seconds = sum(
            duration_sec
            for other_id, other_dt, duration_sec in lesson_times
            if other_id != lesson_id and window_start <= other_dt < dt_obj
        )
        practice_hours[lesson_id] = seconds / 3600.0
    return practice_hours


def _parse_lesson_datetime(lesson_id: str, meta: dict[str, Any]) -> datetime | None:
    for key in ("recorded_at", "created_at", "processed_at"):
        value = meta.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                pass
    try:
        return datetime.strptime(lesson_id.split("_")[0], "%Y%m%d")
    except ValueError:
        return None


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_unique(values: list[str], limit: int) -> list[str]:
    result = []
    seen = set()
    for value in reversed(values):
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
        if len(result) >= limit:
            break
    return result
