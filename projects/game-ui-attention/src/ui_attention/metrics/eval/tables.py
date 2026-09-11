"""报告表格模板骨架（benchmark-protocol.md §7.3 表 A-E，Markdown/JSON 双格式）。

本模块只做"聚合结果 → 表结构/Markdown"的呈现骨架；聚合与 CI 计算由
bootstrap.py 完成、逐图值由 pipeline.py 产出。所有表必须含 U 与 CB 基线行
（缺行 = 报告不完整）；表 E 任一项不同 → 结果不可与旧表同表比较（§8.1）。
所有产物标注"模型预测"与"公开数据评估，非游戏 UI 验证"字样（§7.4）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: 表 A 主表指标列（↑/↓ 标注方向；KL 越低越好）
TABLE_A_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("IG_CB", "↑"),
    ("IG_U", "↑"),
    ("NSS", "↑"),
    ("CC", "↑"),
    ("sAUC", "↑"),
    ("AUC-Judd", "↑"),
    ("KL", "↓"),
)

#: 对外报告红线标注（§7.4/§8.4）
REPORT_DISCLAIMER = (
    "模型预测；公开数据评估，非游戏 UI 验证。任何单一指标数值不得称为“预测准确率”；"
    "引用指标数字必须携带窗口、划分与基线版本。"
)

_MISSING_BASELINE_NOTE = "每个报告表必须含 uniform 与 center_bias 基线行；缺行视为报告不完整（§4.4）"


def _fmt_cell(cell: Mapping[str, Any] | None) -> str:
    if cell is None:
        return "—"
    mean = cell.get("mean")
    if mean is None:
        return "—"
    lo, hi = cell.get("ci_low"), cell.get("ci_high")
    if lo is None or hi is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def _md_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def _check_baseline_rows(rows: Sequence[Mapping[str, Any]], model_key: str = "model") -> list[str]:
    models = {r.get(model_key) for r in rows}
    missing = [name for name in ("uniform", "center_bias") if name not in models]
    return missing


def build_table_a(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """表 A（主表）：模型 × 划分 × 窗口 × 指标；单元格 = mean [95% CI]。

    cells 每行：``{model, split, window, metrics: {name: {mean, ci_low, ci_high}}}``。
    """
    columns = ["model", "split", "window"] + [f"{name} {arrow}" for name, arrow in TABLE_A_METRIC_COLUMNS]
    md_rows: list[list[str]] = []
    for row in cells:
        metrics = row.get("metrics", {})
        md_rows.append(
            [str(row.get("model", "")), str(row.get("split", "")), str(row.get("window", ""))]
            + [_fmt_cell(metrics.get(name)) for name, _ in TABLE_A_METRIC_COLUMNS]
        )
    missing = _check_baseline_rows(cells)
    return {
        "title": "表 A 模型 × 划分 × 窗口 × 指标",
        "columns": columns,
        "rows": [dict(zip(columns, r, strict=True)) for r in md_rows],
        "markdown": _md_table(columns, md_rows),
        "disclaimer": REPORT_DISCLAIMER,
        "missing_baseline_rows": missing,
        "note": _MISSING_BASELINE_NOTE if missing else "",
    }


def build_table_b(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """表 B 配对差值：candidate − baseline 的逐窗口差值均值、CI、胜率、Holm 显著性。"""
    columns = ["model", "baseline", "window", "metric", "mean_diff", "95% CI", "win_rate", "p_adj(Holm)", "significant"]
    md_rows: list[list[str]] = []
    for row in pairs:
        ci_txt = f"[{row.get('ci_low', float('nan')):.4f}, {row.get('ci_high', float('nan')):.4f}]"
        md_rows.append(
            [
                str(row.get("model", "")),
                str(row.get("baseline", "")),
                str(row.get("window", "")),
                str(row.get("metric", "")),
                f"{row.get('mean_diff', float('nan')):.4f}",
                ci_txt,
                f"{row.get('win_rate', float('nan')):.3f}",
                f"{row.get('p_adjusted', float('nan')):.4f}",
                "yes" if row.get("holm_significant") else "no",
            ]
        )
    return {
        "title": "表 B 配对差值（同图配对；两模型 CI 各自不重叠 ≠ 差值显著）",
        "columns": columns,
        "rows": [dict(zip(columns, r, strict=True)) for r in md_rows],
        "markdown": _md_table(columns, md_rows),
        "disclaimer": REPORT_DISCLAIMER,
    }


def build_table_c(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """表 C 类别细分：category × window × 指标（回答"是否只在某类 UI 上有效"）。"""
    columns = ["category", "model", "window"] + [f"{name} {arrow}" for name, arrow in TABLE_A_METRIC_COLUMNS]
    md_rows: list[list[str]] = []
    for row in cells:
        metrics = row.get("metrics", {})
        md_rows.append(
            [str(row.get("category", "")), str(row.get("model", "")), str(row.get("window", ""))]
            + [_fmt_cell(metrics.get(name)) for name, _ in TABLE_A_METRIC_COLUMNS]
        )
    return {
        "title": "表 C 类别细分（分层 bootstrap 口径）",
        "columns": columns,
        "rows": [dict(zip(columns, r, strict=True)) for r in md_rows],
        "markdown": _md_table(columns, md_rows),
        "disclaimer": REPORT_DISCLAIMER,
    }


def build_table_d(audit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """表 D 数据审计：每 (split × window) 的图像数/排除数/参与者覆盖/注视点/重复簇/md5。"""
    columns = [
        "split",
        "window",
        "n_images",
        "excluded_low_fixation",
        "excluded_low_viewers",
        "excluded_empty_samples",
        "viewers_min",
        "viewers_median",
        "viewers_max",
        "n_fixations_total",
        "dup_clusters",
        "dup_clusters_cross_split",
        "md5_ok",
    ]
    md_rows = [[_fmt_value(row.get(c)) for c in columns] for row in audit_rows]
    return {
        "title": "表 D 数据审计（泄漏审计清单 §3.4.5 结果一并记录）",
        "columns": columns,
        "rows": [dict(zip(columns, r, strict=True)) for r in md_rows],
        "markdown": _md_table(columns, md_rows),
    }


def build_table_e(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """表 E 配置指纹：任一项不同 → 结果不可与旧表同表比较（§8.1）。"""
    required_keys = (
        "protocol_version",
        "splits_file_hash",
        "models",
        "preprocessing",
        "numeric_guards",
        "log_base",
        "baseline_versions",
        "eval_code_version",
        "bootstrap",
    )
    missing = [k for k in required_keys if k not in fingerprint]
    md_rows = [[k, _fmt_value(v)] for k, v in sorted(fingerprint.items())]
    return {
        "title": "表 E 配置指纹",
        "columns": ["item", "value"],
        "rows": [{"item": r[0], "value": r[1]} for r in md_rows],
        "markdown": _md_table(["item", "value"], md_rows),
        "missing_required": missing,
        "note": "表 E 任一项不同 → 结果不可与旧表同表比较（§8.1）",
    }


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return "—" if value is None else str(value)
