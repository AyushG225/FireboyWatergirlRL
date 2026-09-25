"""Run the behavior-cloning plus PPO curriculum on the five fixed levels.

Thin entry point; the implementation lives in ``firewater.train_ppo``.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.train_ppo import main

if __name__ == "__main__":
    main()
