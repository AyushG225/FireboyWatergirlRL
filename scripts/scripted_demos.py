"""Generate deterministic successful demonstrations for every level.

Thin entry point; the implementation lives in ``firewater.scripted_demos``.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.scripted_demos import main

if __name__ == "__main__":
    main()
