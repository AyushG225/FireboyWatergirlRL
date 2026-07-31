import os
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from gymnasium.utils.env_checker import check_env

from temple_env import (
    TempleEnv,
    decode_joint_action,
    encode_joint_action,
)
from temple_expert import run_temple_expert
from temple_levels import generate_temple_level


class TempleEnvTests(unittest.TestCase):
    def test_joint_action_round_trip_and_validation(self):
        for fire_action in range(6):
            for water_action in range(6):
                packed = encode_joint_action(fire_action, water_action)
                self.assertEqual(
                    decode_joint_action(packed),
                    (fire_action, water_action),
                )
        with self.assertRaisesRegex(ValueError, "between 0 and 5"):
            encode_joint_action(6, 0)
        with self.assertRaisesRegex(ValueError, "between 0 and 35"):
            decode_joint_action(36)

    def test_environment_follows_gymnasium_contract(self):
        env = TempleEnv(level_seed=100_123)
        check_env(env, skip_render_check=True)
        env.close()

    def test_one_action_moves_both_characters_simultaneously(self):
        env = TempleEnv(level_seed=100_123)
        env.reset(seed=0)
        fire_before = env.fire_x
        water_before = env.water_x

        observation, _, _, _, _ = env.step(encode_joint_action(2, 1))

        self.assertGreater(env.fire_x, fire_before)
        self.assertLess(env.water_x, water_before)
        self.assertTrue(env.observation_space.contains(observation))
        env.close()

    def test_generated_temples_are_reproducible_and_varied(self):
        first = generate_temple_level(100_123)
        repeated = generate_temple_level(100_123)
        different = generate_temple_level(100_124)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first.moving_platforms), 4)
        self.assertEqual(len(first.gates), 3)
        self.assertEqual(len(first.buttons), 6)
        self.assertGreaterEqual(len(first.hazards), 10)

    def test_wrong_element_pool_terminates_episode(self):
        env = TempleEnv(level_seed=100_123)
        env.reset(seed=0)
        water_pool = next(
            hazard for hazard in env.level.hazards if hazard.kind == "water"
        )
        x1, y1, x2, _ = water_pool.rect
        env.fire_x = 0.5 * (x1 + x2)
        env.fire_y = y1

        _, reward, terminated, truncated, info = env.step(
            encode_joint_action(0, 0)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "hazard")
        self.assertLess(reward, 0.0)
        env.close()

    def test_jump_cannot_phase_head_through_platform_above(self):
        env = TempleEnv(level_seed=100_123)
        env.reset(seed=0)
        lower_floor = env.level.floor_y[1]
        ceiling = env.level.platforms[1]
        ceiling_bottom = ceiling[3]
        env.fire_x = 220.0
        env.fire_y = lower_floor
        env.fire_vx = env.fire_vy = 0.0
        env.fire_on_ground = True

        env.step(encode_joint_action(3, 0))
        minimum_head = env.fire_y - 2.0 * env.char_radius
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(
                encode_joint_action(0, 0)
            )
            minimum_head = min(
                minimum_head,
                env.fire_y - 2.0 * env.char_radius,
            )
            self.assertFalse(terminated or truncated)

        self.assertGreaterEqual(minimum_head, ceiling_bottom - 1e-6)
        env.close()

    def test_rgb_render_contains_full_temple_frame(self):
        env = TempleEnv(render_mode="rgb_array", level_seed=100_123)
        env.reset(seed=0)
        frame = env.render()

        self.assertEqual(frame.shape, (720, 960, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(np.unique(frame.reshape(-1, 3), axis=0).shape[0], 50)
        env.close()

    def test_expert_solves_unseen_temples_with_joint_movement(self):
        for layout_seed in range(100_123, 100_126):
            with self.subTest(layout_seed=layout_seed):
                env = TempleEnv(level_seed=layout_seed)
                env.reset(seed=0)
                run = run_temple_expert(env)
                self.assertEqual(run.reason, "success")
                self.assertGreater(run.simultaneous_frames, 300)
                self.assertLess(len(run.actions), env.max_steps)
                env.close()


if __name__ == "__main__":
    unittest.main()
