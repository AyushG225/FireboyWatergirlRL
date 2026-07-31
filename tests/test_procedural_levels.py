import unittest

from procedural_levels import generate_level, held_out_seeds, training_seeds


class ProceduralLevelTests(unittest.TestCase):
    def test_generation_is_deterministic_and_varied(self):
        first = generate_level(42)
        repeated = generate_level(42)
        different = generate_level(43)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertLessEqual(len(first.platforms), 8)

    def test_training_and_held_out_seed_ranges_are_disjoint(self):
        train = set(training_seeds(1_000))
        held_out = set(held_out_seeds(1_000))

        self.assertTrue(train.isdisjoint(held_out))

    def test_invalid_difficulty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "difficulty"):
            generate_level(0, difficulty=3)
