import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firewater.rollout_trained import checkpoint_exists, preferred_model


class RolloutTests(unittest.TestCase):
    def test_checkpoint_exists_accepts_sb3_implicit_zip_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent"
            checkpoint.with_suffix(".zip").touch()

            self.assertTrue(checkpoint_exists(checkpoint))

    def test_preferred_model_uses_first_existing_checkpoint(self):
        candidates = (Path("first"), Path("second"))

        with patch("firewater.rollout_trained.PREFERRED_MODELS", candidates), patch(
            "firewater.rollout_trained.checkpoint_exists",
            side_effect=lambda path: path == candidates[1],
        ):
            self.assertEqual(preferred_model(), candidates[1])


if __name__ == "__main__":
    unittest.main()
