"""ui_attention：游戏 UI 视觉注意力评估最小工具。

模块所有权（implementation-plan.md §阶段1-3 所有权表）：
- C1：__init__/cli/errors/imaging、contracts/、aoi/、metrics/（含 eval/）、pyproject、lock、fixtures
- C2：backends/（经 contracts/backend.py 冻结接口协作）
- C3：report/（经 cli.py 动态导入协作）

数值边界：模型输出是预测分布，不是真实眼动或点击率（technical-design.md §1）。
"""

__version__ = "0.1.0"
