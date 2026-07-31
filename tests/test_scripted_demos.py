import unittest

import numpy as np

from scripted_demos import generate_scripted_demo


class ScriptedDemonstrationTests(unittest.TestCase):
    def test_scripted_controller_solves_every_level(self):
        for level_id in range(5):
            with self.subTest(level_id=level_id):
                demo = generate_scripted_demo(level_id)

                self.assertEqual(demo.reason, "success")
                self.assertLessEqual(len(demo.actions), 300)
                self.assertEqual(demo.observations.shape, (len(demo.actions), 56))
                self.assertEqual(demo.observations.dtype, np.float32)
                self.assertEqual(demo.actions.dtype, np.int64)
                self.assertTrue(np.all((0 <= demo.actions) & (demo.actions < 7)))
                self.assertGreater(float(demo.rewards.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
