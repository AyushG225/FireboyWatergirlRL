"""Plan and verify a route through a procedurally generated level.

Thin entry point; the implementation lives in ``firewater.generalized_planner``.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.generalized_planner import main

if __name__ == "__main__":
    main()
