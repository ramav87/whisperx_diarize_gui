import unittest

from diarize_gui.metrics.context_adjusted import (
    build_context_metrics,
    classify_context,
    compute_adjusted_grammar_score,
    compute_automaticity_gap,
    compute_conceptual_load_score,
    compute_cognitive_load_adjusted_wpm,
    compute_fluency_under_load,
)


class ContextAdjustedMetricsTests(unittest.TestCase):
    def test_adjusted_grammar_acceptance_example(self):
        score = compute_adjusted_grammar_score(72, 0.2, 5, 2)
        self.assertAlmostEqual(score, 82.2)

    def test_missing_optional_values_do_not_change_raw_grammar(self):
        self.assertEqual(compute_adjusted_grammar_score(72, None, None, None), 72)

    def test_adjusted_grammar_clamps_above_100(self):
        self.assertEqual(compute_adjusted_grammar_score(99, 0, 5, 3), 100)

    def test_cold_and_hard_classification(self):
        labels = classify_context(0.5, 4, 4)
        self.assertTrue(labels["cold_session"])
        self.assertTrue(labels["hard_topic"])
        self.assertTrue(labels["high_idea_density"])

    def test_wpm_acceptance_examples(self):
        self.assertEqual(compute_cognitive_load_adjusted_wpm(94, 5, 5), 110)
        self.assertAlmostEqual(compute_fluency_under_load(94, 5, 5, 3.2), 100.4)

    def test_missing_load_context_does_not_emit_fake_adjusted_wpm(self):
        metrics = build_context_metrics(raw_grammar_score=80, raw_wpm=72)
        self.assertEqual(metrics["raw_wpm"], 72)
        self.assertNotIn("difficulty_adjusted_wpm", metrics)
        self.assertNotIn("cognitive_load_adjusted_wpm", metrics)
        self.assertNotIn("fluency_under_load", metrics)

    def test_advanced_load_context_changes_adjusted_metrics(self):
        load = compute_conceptual_load_score(
            topic_difficulty=5,
            idea_density=5,
            abstraction_level=9,
            cognitive_branching=8,
            technical_density=10,
            discourse_depth=8,
            lexical_retrieval_pressure=9,
        )
        self.assertGreater(load, 9)

        metrics = build_context_metrics(
            raw_grammar_score=80,
            raw_wpm=75,
            topic_difficulty=5,
            idea_density=5,
            abstraction_level=9,
            cognitive_branching=8,
            technical_density=10,
            discourse_depth=8,
            lexical_retrieval_pressure=9,
        )
        self.assertGreater(metrics["adjusted_grammar_score"], 86)
        self.assertIn("effective_fluency_score", metrics)
        self.assertIn("complexity_resilience_score", metrics)

    def test_automaticity_gap_mixed_sessions(self):
        sessions = [
            {"grammar_score": 70, "context_metrics": {"practice_hours_last_7_days": 0.2}},
            {"grammar_score": 80, "context_metrics": {"practice_hours_last_7_days": 2.0}},
            {"grammar_score": 74, "context_metrics": {"practice_hours_last_7_days": 0.5}},
            {"grammar_score": 86, "context_metrics": {"practice_hours_last_7_days": 3.0}},
        ]
        gap = compute_automaticity_gap(sessions, window=10)
        self.assertEqual(gap["warm_session_count"], 2)
        self.assertEqual(gap["cold_session_count"], 2)
        self.assertAlmostEqual(gap["warm_adjusted_or_raw_grammar_avg"], 83)
        self.assertAlmostEqual(gap["cold_adjusted_or_raw_grammar_avg"], 72)
        self.assertAlmostEqual(gap["automaticity_gap"], 11)

    def test_old_lesson_record_does_not_crash(self):
        metrics = build_context_metrics(raw_grammar_score=None, raw_wpm=None)
        self.assertNotIn("adjusted_grammar_score", metrics)
        gap = compute_automaticity_gap([{"grammar_score": 90}], window=10)
        self.assertIsNone(gap["automaticity_gap"])


if __name__ == "__main__":
    unittest.main()
