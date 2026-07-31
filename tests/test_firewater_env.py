import os
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from gymnasium.utils.env_checker import check_env

from firewater_env import FireWaterEnv


class FireWaterEnvTests(unittest.TestCase):
    def test_all_levels_follow_gymnasium_contract(self):
        for observation_mode in (
            "legacy",
            "enhanced_v1",
            "enhanced",
            "generalized",
        ):
            for level_id in range(5):
                with self.subTest(
                    level_id=level_id,
                    observation_mode=observation_mode,
                ):
                    env = FireWaterEnv(
                        level_id=level_id,
                        observation_mode=observation_mode,
                    )
                    check_env(env, skip_render_check=True)
                    env.close()

    def test_random_steps_stay_inside_observation_space(self):
        for level_id in range(5):
            with self.subTest(level_id=level_id):
                env = FireWaterEnv(level_id=level_id)
                observation, _ = env.reset(seed=level_id)
                self.assertTrue(env.observation_space.contains(observation))

                for _ in range(50):
                    observation, reward, terminated, truncated, info = env.step(
                        env.action_space.sample()
                    )
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertTrue(np.isfinite(reward))
                    self.assertIn("reason", info)
                    if terminated or truncated:
                        break
                env.close()

    def test_both_doors_end_the_episode_successfully(self):
        env = FireWaterEnv(level_id=0)
        env.reset()
        env.fire_x, env.fire_y = env.fire_goal_x, env.fire_goal_y
        env.water_x, env.water_y = env.water_goal_x, env.water_goal_y

        _, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["reason"], "success")
        env.close()

    def test_wrong_pool_ends_the_episode(self):
        env = FireWaterEnv(level_id=0)
        env.reset()
        water_x1, _, water_x2, _ = env.water_rect
        env.fire_x = (water_x1 + water_x2) / 2

        _, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0.0)
        self.assertGreater(reward, -20.0)
        self.assertEqual(info["reason"], "hazard")
        env.close()

    def test_partial_solution_then_death_is_strongly_negative(self):
        env = FireWaterEnv(level_id=2)
        env.reset()
        env.fire_x, env.fire_y = env.fire_goal_x, env.fire_goal_y
        env.fire_at_goal = True
        lava_x1, _, lava_x2, _ = env.lava_rect
        env.water_x = (lava_x1 + lava_x2) / 2

        _, reward, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(info["reason"], "hazard")
        self.assertLess(reward, -20.0)
        env.close()

    def test_step_limit_truncates_the_episode(self):
        env = FireWaterEnv(level_id=0)
        env.max_steps = 1
        env.reset()

        _, _, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "timeout")
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            env.step(0)

        observation, _ = env.reset()
        self.assertTrue(env.observation_space.contains(observation))
        env.close()

    def test_team_distance_includes_character_after_first_goal(self):
        env = FireWaterEnv(level_id=0)
        env.reset()
        env.fire_x, env.fire_y = env.fire_goal_x, env.fire_goal_y
        env.fire_at_goal = True

        before = env._compute_team_goal_dist()
        env.step(4)
        after = env._compute_team_goal_dist()

        self.assertLess(after, before)
        env.close()

    def test_rgb_array_rendering_is_headless_and_repeatable(self):
        env = FireWaterEnv(render_mode="rgb_array", level_id=2)
        env.reset()

        first_frame = env.render()
        env.step(0)
        second_frame = env.render()

        self.assertEqual(first_frame.shape, (env.H, env.W, 3))
        self.assertEqual(first_frame.dtype, np.uint8)
        self.assertEqual(second_frame.shape, first_frame.shape)
        env.close()

    def test_invalid_configuration_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported render_mode"):
            FireWaterEnv(render_mode="video")
        with self.assertRaisesRegex(ValueError, "observation_mode"):
            FireWaterEnv(observation_mode="pixels")

        env = FireWaterEnv(level_id=5)
        with self.assertRaisesRegex(ValueError, "Unknown level_id"):
            env.reset()

    def test_enhanced_observation_exposes_platform_layout(self):
        flat = FireWaterEnv(level_id=0, observation_mode="enhanced")
        platformed = FireWaterEnv(level_id=4, observation_mode="enhanced")
        flat_obs, _ = flat.reset()
        platform_obs, _ = platformed.reset()

        self.assertEqual(flat_obs.shape, (56,))
        self.assertEqual(platform_obs.shape, (56,))
        self.assertFalse(np.array_equal(flat_obs[19:], platform_obs[19:]))
        flat.close()
        platformed.close()

    def test_enhanced_observation_exposes_hazard_boundaries(self):
        env = FireWaterEnv(level_id=2, observation_mode="enhanced")
        observation, _ = env.reset()
        expected_lava_x1 = (env.lava_rect[0] / env.W) * 2.0 - 1.0
        expected_water_x2 = (env.water_rect[2] / env.W) * 2.0 - 1.0

        self.assertAlmostEqual(float(observation[48]), expected_lava_x1)
        self.assertAlmostEqual(float(observation[54]), expected_water_x2)
        env.close()

    def test_procedural_seed_is_reproducible_and_has_no_level_identity(self):
        first = FireWaterEnv(
            procedural=True,
            level_seed=100_000,
            observation_mode="generalized",
        )
        second = FireWaterEnv(
            procedural=True,
            level_seed=100_000,
            observation_mode="generalized",
        )
        first_observation, first_info = first.reset(seed=1)
        second_observation, second_info = second.reset(seed=999)

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first_observation.shape, (64,))
        self.assertEqual(first_info["layout_seed"], 100_000)
        self.assertEqual(second_info["layout_seed"], 100_000)
        self.assertEqual(first_info["level_id"], -1)
        self.assertTrue(first_info["procedural"])
        first.close()
        second.close()

    def test_random_procedural_env_resamples_layouts(self):
        env = FireWaterEnv(
            procedural=True,
            observation_mode="generalized",
            procedural_seed_range=(0, 100_000),
        )
        _, first_info = env.reset(seed=123)
        _, second_info = env.reset()

        self.assertNotEqual(first_info["layout_seed"], second_info["layout_seed"])
        self.assertLess(first_info["layout_seed"], 100_000)
        self.assertLess(second_info["layout_seed"], 100_000)
        env.close()


if __name__ == "__main__":
    unittest.main()
