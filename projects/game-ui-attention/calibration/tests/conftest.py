"""pytest 夹具：把 calibration/tools 加入 sys.path，供测试直接 import consistency / make_study_package。

两个工具模块为自包含实现（仅依赖 stdlib + numpy/pillow），不依赖 src 包，
因此无需 PYTHONPATH=src 也能被本目录测试导入；主项目测试仍按 pyproject 的 pythonpath=src 运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
