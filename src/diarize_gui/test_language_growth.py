import unittest

from diarize_gui.metrics.language_growth import (
    compute_language_growth_metrics,
    moving_average_type_token_ratio,
    tokenize,
)


class LanguageGrowthMetricTests(unittest.TestCase):
    def test_tokenizer_preserves_spanish_letters(self):
        self.assertEqual(tokenize("¡Qué hipótesis más extraña!"), ["qué", "hipótesis", "más", "extraña"])

    def test_mattr_is_length_normalized(self):
        varied = [f"palabra{index}" for index in range(60)]
        self.assertEqual(moving_average_type_token_ratio(varied, 50), 1.0)
        self.assertAlmostEqual(moving_average_type_token_ratio(["uno", "uno", "dos"], 50), 2 / 3)

    def test_metrics_use_only_explicit_student_segments(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "aunque porque además tutor"},
            {"speaker": "SPEAKER_01", "text": "Aunque era difícil, seguí hablando. Además expliqué mi hipótesis."},
        ]
        metrics = compute_language_growth_metrics(segments, ["SPEAKER_01"])
        self.assertEqual(metrics["speaker_scope"], "explicit")
        self.assertGreater(metrics["token_count"], 5)
        self.assertGreaterEqual(metrics["connector_types"], 2)
        self.assertNotIn("tutor", [item["word"] for item in metrics["top_content_words"]])

    def test_missing_assignment_is_labeled_as_inferred(self):
        metrics = compute_language_growth_metrics(
            [{"speaker": "SPEAKER_01", "text": "Estoy explicando una idea compleja."}],
            [],
        )
        self.assertEqual(metrics["speaker_scope"], "inferred_speaker_01")


if __name__ == "__main__":
    unittest.main()
