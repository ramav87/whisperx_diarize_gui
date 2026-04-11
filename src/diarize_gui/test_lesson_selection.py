import json
import os
import tempfile
import unittest

from diarize_gui.lesson_selection import select_pending_ai_lesson_dirs


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
                        json.dump({"grammar_score": 90}, f)

            make_lesson("20260410_030000", processed=False)
            make_lesson("20260410_020000", processed=False)
            make_lesson("20260410_010000", processed=True)
            make_lesson("20260409_230000", processed=False)

            pending = [os.path.basename(path) for path in select_pending_ai_lesson_dirs(lessons_dir)]

        self.assertEqual(pending, ["20260410_030000", "20260410_020000"])


if __name__ == "__main__":
    unittest.main()
