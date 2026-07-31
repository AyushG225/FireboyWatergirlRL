"""Make the repository root importable when a script is run directly.

Running ``python scripts/watch_temple.py`` puts ``scripts/`` on ``sys.path``
rather than the repository root, so ``import firewater`` would fail. Importing
this module first prepends the root and keeps the scripts runnable straight
from a fresh clone with no install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
