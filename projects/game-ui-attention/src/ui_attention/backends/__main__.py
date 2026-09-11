"""Explicit weight install step: ``python -m ui_attention.backends``.

Delegates to :func:`ui_attention.backends.weights._main` — downloads ONLY the
registered foveacast 3s FP16 artifact into ``model-cache/foveacast/`` with
sha256 verification (single small file, approved by 二级总控; 1s/7s
window-sensitivity weights pending approval, nothing else is ever fetched).
Exit 0 on success / cache hit, 3 on structured failure. Using the package
``__main__`` avoids the runpy double-import warning that
``python -m ui_attention.backends.weights`` would emit (backends/__init__
already imports the weights module).
"""

from __future__ import annotations

from .weights import _main

if __name__ == "__main__":
    raise SystemExit(_main())
