import unittest
from unittest.mock import patch

from firewater.evaluation import LevelEvaluation, evaluate_model, format_evaluation


class ConstantPolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, deterministic=True):
        return self.action, None


class EvaluationTests(unittest.TestCase):
    @patch("firewater.evaluation.set_random_seed")
    def test_evaluation_seeds_policy_sampling(self, set_seed):
        model = ConstantPolicy(action=0)

        evaluate_model(model, levels=[0], episodes=1, seed=17)

        set_seed.assert_called_once_with(17)

    def test_evaluate_model_tracks_terminal_reasons(self):
        results = evaluate_model(ConstantPolicy(0), levels=[0], episodes=2)
        level = results[0]

        self.assertEqual(level.episodes, 2)
        self.assertEqual(level.timeouts, 2)
        self.assertEqual(level.successes, 0)
        self.assertEqual(level.hazards, 0)
        self.assertEqual(level.timeout_rate, 1.0)
        self.assertEqual(level.mean_length, 300.0)

    def test_format_evaluation_contains_rates(self):
        result = LevelEvaluation(
            level_id=2,
            episodes=4,
            successes=3,
            hazards=1,
            timeouts=0,
            mean_return=42.5,
            std_return=1.0,
            mean_length=100.0,
        )

        rendered = format_evaluation({2: result})

        self.assertIn("75.0%", rendered)
        self.assertIn("25.0%", rendered)
        self.assertIn("42.500", rendered)

    def test_invalid_episode_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            evaluate_model(ConstantPolicy(0), levels=[0], episodes=0)


if __name__ == "__main__":
    unittest.main()
