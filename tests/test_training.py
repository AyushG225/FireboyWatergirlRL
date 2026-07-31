import tempfile
import unittest
from pathlib import Path

import numpy as np

from firewater.firewater_env import FireWaterEnv
from firewater.train_ppo import CURRICULUM, assign_levels, load_demos


class TrainingUtilityTests(unittest.TestCase):
    def test_assign_levels_scales_curriculum_mix(self):
        mix = (0, 1, 2, 2, 3, 3, 4, 4)

        self.assertEqual(assign_levels(mix, 8), list(mix))
        self.assertEqual(assign_levels(mix, 5), [0, 1, 2, 3, 4])
        self.assertEqual(assign_levels(mix, 4), [0, 2, 3, 4])
        self.assertEqual(assign_levels(mix, 1), [0])

    def test_assign_levels_preserves_all_levels_in_hard_weighting(self):
        hard_mix = (0, 1, 2, 3, 3, 4, 4, 4)

        self.assertEqual(set(assign_levels(hard_mix, 5)), set(range(5)))

    def test_assign_levels_rejects_empty_configuration(self):
        with self.assertRaisesRegex(ValueError, "n_envs"):
            assign_levels((0,), 0)
        with self.assertRaisesRegex(ValueError, "level_mix"):
            assign_levels((), 1)

    def test_late_curriculum_refines_hard_levels_before_consolidation(self):
        self.assertEqual(CURRICULUM[-2].slug, "hard_levels")
        self.assertGreater(CURRICULUM[-2].level_mix.count(3), 1)
        self.assertGreater(CURRICULUM[-2].level_mix.count(4), 1)
        self.assertGreater(CURRICULUM[-2].entropy_coef, 0.0)
        self.assertEqual(set(CURRICULUM[-1].level_mix), set(range(5)))
        self.assertEqual(CURRICULUM[-1].entropy_coef, 0.0)

    def test_load_demos_validates_and_concatenates_files(self):
        env = FireWaterEnv()
        observation, _ = env.reset()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                np.savez(
                    root / f"demo_level0_{index}.npz",
                    obs=np.stack([observation, observation, observation]),
                    acts=np.array([0, 0, 1], dtype=np.int64),
                )

            observations, actions = load_demos(
                str(root / "demo_*.npz"),
                observation_mode="legacy",
            )

        self.assertEqual(observations.shape, (2, 19))
        self.assertEqual(actions.tolist(), [1, 1])
        env.close()

    def test_load_demos_skips_invalid_actions(self):
        env = FireWaterEnv()
        observation, _ = env.reset()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo_bad.npz"
            np.savez(
                path,
                obs=np.stack([observation]),
                acts=np.array([99], dtype=np.int64),
            )
            observations, actions = load_demos(
                str(path),
                observation_mode="legacy",
            )

        self.assertIsNone(observations)
        self.assertIsNone(actions)
        env.close()

    def test_load_demos_replays_legacy_recording_for_enhanced_training(self):
        env = FireWaterEnv(level_id=2, observation_mode="legacy")
        observation, _ = env.reset()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo_level2_old.npz"
            np.savez(
                path,
                obs=np.stack([observation, observation]),
                acts=np.array([0, 1], dtype=np.int64),
            )
            observations, actions = load_demos(
                str(path),
                observation_mode="enhanced",
            )

        self.assertEqual(observations.shape, (1, 56))
        self.assertEqual(actions.tolist(), [1])
        self.assertEqual(observations[0, 23], 0.0)  # normalized level 2 id
        env.close()


if __name__ == "__main__":
    unittest.main()
