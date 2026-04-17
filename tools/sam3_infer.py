from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.sam_infer_core import main


if __name__ == "__main__":
    raise SystemExit(main())
