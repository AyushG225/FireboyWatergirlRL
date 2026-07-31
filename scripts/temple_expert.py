"""Benchmark the closed-loop temple expert on held-out seeds.

Thin entry point; the implementation lives in ``firewater.temple_expert``.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.temple_expert import main

if __name__ == "__main__":
    main()
