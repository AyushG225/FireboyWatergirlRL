import unittest

from firewater_env import FireWaterEnv
from safety_shield import shield_action


class SafetyShieldTests(unittest.TestCase):
    def test_imminent_wrong_pool_entry_forces_jump(self):
        env = FireWaterEnv(level_id=0)
        env.reset()
        water_x1, _, _, _ = env.water_rect
        env.fire_x = water_x1 - 60.0
        env.fire_vx = env.move_speed
        env.fire_on_ground = True

        self.assertEqual(shield_action(env, 2), 3)
        env.close()

    def test_safe_action_is_not_changed(self):
        env = FireWaterEnv(level_id=0)
        env.reset()

        self.assertEqual(shield_action(env, 2), 2)
        env.close()

    def test_invalid_margin_is_rejected(self):
        env = FireWaterEnv(level_id=0)
        env.reset()
        with self.assertRaisesRegex(ValueError, "positive"):
            shield_action(env, 0, margin=0)
        env.close()


if __name__ == "__main__":
    unittest.main()
