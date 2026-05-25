import json
import os
import tempfile
import unittest

from diarize_gui.lesson_selection import (
    ai_stats_complete,
    select_all_incomplete_ai_lesson_dirs,
    select_pending_ai_lesson_dirs,
)


def complete_ai_stats(grammar_score: int = 90) -> dict:
    return {
        "grammar_score": grammar_score,
        "context_metrics": {
            "raw_grammar_score": grammar_score,
            "adjusted_grammar_score": grammar_score,
            "topic_difficulty": 3,
            "idea_density": 3,
            "abstraction_level": 4,
            "cognitive_branching": 4,
            "technical_density": 2,
            "discourse_depth": 4,
            "lexical_retrieval_pressure": 3,
            "conceptual_load_score": 6,
        },
    }


class LessonSelectionTests(unittest.TestCase):
    def test_selects_newest_contiguous_unprocessed_lessons_only(self):
        with tempfile.TemporaryDirectory() as root:
            lessons_dir = os.path.join(root, "lessons")
            os.makedirs(lessons_dir)

            def make_lesson(name: str, processed: bool) -> None:
                lesson_dir = os.path.join(lessons_dir, name)
                os.makedirs(lesson_dir)
                with open(os.path.join(lesson_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({}, f)
                with open(os.path.join(lesson_dir, "segments.json"), "w", encoding="utf-8") as f:
                    json.dump([], f)
                if processed:
                    with open(os.path.join(lesson_dir, "ai_stats.json"), "w", encoding="utf-8") as f:
                        json.dump(complete_ai_stats(), f)

            make_lesson("20260410_030000", processed=False)
            make_lesson("20260410_020000", processed=False)
            make_lesson("20260410_010000", processed=True)
            make_lesson("20260409_230000", processed=False)

            pending = [os.path.basename(path) for path in select_pending_ai_lesson_dirs(lessons_dir)]

        self.assertEqual(pending, ["20260410_030000", "20260410_020000"])

    def test_partial_ai_stats_still_count_as_pending(self):
        with tempfile.TemporaryDirectory() as root:
            lessons_dir = os.path.join(root, "lessons")
            os.makedirs(lessons_dir)

            def make_lesson(name: str, stats: dict) -> None:
                lesson_dir = os.path.join(lessons_dir, name)
                os.makedirs(lesson_dir)
                with open(os.path.join(lesson_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({}, f)
                with open(os.path.join(lesson_dir, "segments.json"), "w", encoding="utf-8") as f:
                    json.dump([], f)
                with open(os.path.join(lesson_dir, "ai_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f)

            make_lesson("20260410_030000", {"grammar_score": 90})
            make_lesson(
                "20260410_020000",
                {
                    **complete_ai_stats(85),
                },
            )

            pending = [os.path.basename(path) for path in select_pending_ai_lesson_dirs(lessons_dir)]

        self.assertEqual(pending, ["20260410_030000"])

    def test_ai_stats_complete_requires_context_fields(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "ai_stats.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"grammar_score": 90}, f)

            self.assertFalse(ai_stats_complete(path))

            with open(path, "w", encoding="utf-8") as f:
                json.dump(complete_ai_stats(), f)

            self.assertTrue(ai_stats_complete(path))

    def test_select_all_incomplete_includes_older_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            lessons_dir = os.path.join(root, "lessons")
            os.makedirs(lessons_dir)

            complete_stats = complete_ai_stats()

            def make_lesson(name: str, stats) -> None:
                lesson_dir = os.path.join(lessons_dir, name)
                os.makedirs(lesson_dir)
                with open(os.path.join(lesson_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({}, f)
                with open(os.path.join(lesson_dir, "segments.json"), "w", encoding="utf-8") as f:
                    json.dump([], f)
                if stats is not None:
                    with open(os.path.join(lesson_dir, "ai_stats.json"), "w", encoding="utf-8") as f:
                        json.dump(stats, f)

            make_lesson("20260410_030000", complete_stats)
            make_lesson("20260410_020000", {"grammar_score": 75})
            make_lesson("20260410_010000", complete_stats)
            make_lesson("20260409_230000", None)

            pending = [os.path.basename(path) for path in select_all_incomplete_ai_lesson_dirs(lessons_dir)]

        self.assertEqual(pending, ["20260410_020000", "20260409_230000"])


if __name__ == "__main__":
    unittest.main()
