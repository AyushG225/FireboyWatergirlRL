"""Play back a trained checkpoint on a fixed level.

Thin entry point; the implementation lives in ``firewater.rollout_trained``.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.rollout_trained import main

if __name__ == "__main__":
    main()
