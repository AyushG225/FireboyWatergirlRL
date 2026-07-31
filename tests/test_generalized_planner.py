import unittest

from firewater_env import FireWaterEnv
from generalized_planner import plan_level


class GeneralizedPlannerTests(unittest.TestCase):
    def test_planner_solves_held_out_layouts(self):
        for seed in (100_000, 100_001, 100_002):
            with self.subTest(seed=seed):
                env = FireWaterEnv(
                    procedural=True,
                    level_seed=seed,
                    observation_mode="generalized",
                )
                env.reset(seed=0)
                try:
                    result = plan_level(env)
                    self.assertLessEqual(len(result.actions), env.max_steps)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
