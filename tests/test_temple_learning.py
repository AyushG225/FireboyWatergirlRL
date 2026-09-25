import os
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from firewater.firewater_env import FireWaterEnv
from firewater.temple_env import TempleEnv
from firewater.temple_evaluation import evaluate_temple_expert
from firewater.temple_expert import TempleExpert
from firewater.train_ppo import behavior_clone


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


class ExpertLabelTests(unittest.TestCase):
    def test_stateless_expert_matches_stateful_expert_on_its_own_path(self):
        env = TempleEnv(level_seed=90_004)
        env.reset(seed=0)
        stateful = TempleExpert()
        stateless = TempleExpert(stateless=True)
        for _ in range(env.max_steps):
            action = stateful.action(env)
            self.assertEqual(action, stateless.action(env))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        self.assertEqual(info["reason"], "success")

    def test_stage_from_height_counts_floors_reached(self):
        env = TempleEnv(level_seed=90_005)
        env.reset(seed=0)
        for floor_index, floor_y in enumerate(env.level.floor_y):
            env.fire_y = floor_y
            self.assertEqual(TempleExpert.stage_from_height(env, "fire"), floor_index)


class TempleEvaluationTests(unittest.TestCase):
    def test_batched_expert_evaluation_is_consistent(self):
        result = evaluate_temple_expert([90_010, 90_011, 90_012])
        self.assertEqual(result.successes, 3)
        self.assertEqual(result.hazards + result.timeouts, 0)
        for episode in result.episodes:
            self.assertLessEqual(episode.both_moving_frames, episode.both_command_frames)
            self.assertLessEqual(episode.both_command_frames, episode.steps)
        self.assertGreater(result.both_moving_share, 0.0)
        self.assertLess(result.both_moving_share, 1.0)


class WeightedCloningTests(unittest.TestCase):
    def test_sample_weights_shift_the_cloned_policy(self):
        env = DummyVecEnv([lambda: FireWaterEnv()])
        observations = np.zeros((200, 19), dtype=np.float32)
        actions = np.array([0] * 180 + [3] * 20)

        def cloned_action(weights):
            model = PPO(
                "MlpPolicy", env, policy_kwargs={"net_arch": [16]}, seed=0, device="cpu"
            )
            behavior_clone(
                model,
                observations,
                actions,
                n_epochs=30,
                batch_size=50,
                learning_rate=1e-2,
                sample_weights=weights,
            )
            return int(model.predict(observations[0], deterministic=True)[0])

        self.assertEqual(cloned_action(None), 0)
        self.assertEqual(cloned_action(np.where(actions == 3, 20.0, 1.0)), 3)
        with self.assertRaisesRegex(ValueError, "sample_weights"):
            model = PPO("MlpPolicy", env, seed=0, device="cpu")
            behavior_clone(model, observations, actions, sample_weights=np.ones(3))
        env.close()


if __name__ == "__main__":
    unittest.main()
