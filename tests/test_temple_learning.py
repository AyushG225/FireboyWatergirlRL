import os
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from gymnasium.utils.env_checker import check_env

from firewater.temple_env import TempleEnv


class ObservationModeTests(unittest.TestCase):
    def test_modes_share_absolute_prefix(self):
        absolute = TempleEnv(level_seed=90_001, observation_mode="absolute")
        egocentric = TempleEnv(level_seed=90_001, observation_mode="egocentric")
        short, _ = absolute.reset(seed=0)
        full, _ = egocentric.reset(seed=0)
        self.assertEqual(short.shape, (250,))
        self.assertEqual(full.shape, (270,))
        np.testing.assert_array_equal(short, full[:250])
        self.assertEqual(TempleEnv.mode_for_size(250), "absolute")
        self.assertEqual(TempleEnv.mode_for_size(270), "egocentric")
        with self.assertRaises(ValueError):
            TempleEnv.mode_for_size(64)
        with self.assertRaises(ValueError):
            TempleEnv(observation_mode="pixels")

    def test_absolute_mode_follows_gymnasium_contract(self):
        env = TempleEnv(level_seed=90_002, observation_mode="absolute")
        check_env(env, skip_render_check=True)

    def test_pool_sensor_reads_distance_to_deadly_pool_only(self):
        env = TempleEnv(level_seed=90_003)
        env.reset(seed=0)
        water_pool = next(
            hazard for hazard in env.level.hazards if hazard.kind == "water"
        )
        x1, y1, _, _ = water_pool.rect
        env.fire_x, env.fire_y = x1 - 50.0, y1 + 3.0
        env.water_x, env.water_y = x1 - 50.0, y1 + 3.0
        features = env._egocentric_features()
        # Fire dies in water: 50 px of a 200 px sensor range.
        self.assertAlmostEqual(features[0], 0.25, places=5)
        # Water is immune to water pools, so its right sensor skips this one.
        self.assertGreater(features[10], 0.25)


if __name__ == "__main__":
    unittest.main()
