"""comparison v1 schema 与兼容性/配对逻辑（data-contract.md §6、technical-design.md §9）。

对比要求模型、权重、预处理、中心偏置、观看配置、指标版本兼容；第一版还要求相同画布尺寸。
通过稳定 AOI ID 配对，位置和大小可以改变，但必须提供各自的有效边界。
新增或删除区域单独列出，不编造缺失一侧的 0 值；不匹配区域不计算差值。
预处理、模型、先验、画布尺寸等不兼容时拒绝正式比较，保留原因（退出码 6）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ErrorCode, UiAttentionError
from ..metrics.regions import compare_region_rows
from .analysis import AnalysisRecord

COMPARISON_SCHEMA_VERSION = "game-ui-attention-comparison/v1"

# 兼容性维度（technical-design §9；画布尺寸第一版强制相同）
_COMPATIBILITY_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("model.backend_id", "模型后端 ID"),
    ("model.version", "后端实现版本"),
    ("model.weights_sha256", "全部权重哈希"),
    ("model.code_version", "后端代码版本"),
    ("profile.config_hash", "预处理/中心偏置/观看条件合成哈希"),
    ("input.width", "画布宽度（第一版要求相同画布尺寸）"),
    ("input.height", "画布高度（第一版要求相同画布尺寸）"),
    ("metrics_version", "AOI 指标口径版本"),
    ("evidence_type", "证据类型"),
)


def _get_path(record: AnalysisRecord, dotted: str) -> Any:
    obj: Any = record.to_dict()
    for part in dotted.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


@dataclass(frozen=True)
class CompatibilityReport:
    """配置兼容性检查结果；mismatches 保留原因（不兼容时结构化退出，不降级为文字警告）。"""

    compatible: bool
    mismatches: tuple[dict[str, Any], ...] = ()
    canvas_note: str = ""
    exploratory: bool = False
    exploratory_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "mismatches": [dict(m) for m in self.mismatches],
            "canvas_note": self.canvas_note,
            "exploratory": self.exploratory,
            "exploratory_reasons": list(self.exploratory_reasons),
        }


def check_compatibility(before: AnalysisRecord, after: AnalysisRecord) -> CompatibilityReport:
    """逐维度校验 A/B 配置兼容性；任一维度不同 → 不兼容并保留原因。"""
    mismatches: list[dict[str, Any]] = []
    for dotted, description in _COMPATIBILITY_DIMENSIONS:
        b = _get_path(before, dotted)
        a = _get_path(after, dotted)
        if b != a:
            mismatches.append({"field": dotted, "description": description, "before": b, "after": a})

    same_canvas = (before.input.get("width"), before.input.get("height")) == (
        after.input.get("width"),
        after.input.get("height"),
    )
    canvas_note = (
        "画布尺寸相同，满足第一版正式比较条件"
        if same_canvas
        else "画布尺寸不同：第一版拒绝正式比较（不能依靠“都叫 attention”跨配置直接比较）"
    )

    # 画面可比性说明：画面状态、任务或内容明显不同 → 标为探索性比较，不作修改的因果结论。
    exploratory_reasons: list[str] = []
    if before.input.get("screen_type") != after.input.get("screen_type"):
        exploratory_reasons.append(
            f"画面类型不同（{before.input.get('screen_type')!r} → {after.input.get('screen_type')!r}）"
        )
    if before.player_goal != after.player_goal:
        exploratory_reasons.append("玩家目标不同或缺失一侧目标")

    return CompatibilityReport(
        compatible=not mismatches,
        mismatches=tuple(mismatches),
        canvas_note=canvas_note,
        exploratory=bool(exploratory_reasons),
        exploratory_reasons=tuple(exploratory_reasons),
    )


@dataclass(frozen=True)
class ComparisonRecord:
    """comparison.json 的内存形态（data-contract §6）。"""

    before_analysis_id: str
    after_analysis_id: str
    compatibility: CompatibilityReport
    matched: tuple[dict[str, Any], ...]
    added: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]
    schema_version: str = COMPARISON_SCHEMA_VERSION
    created_at_utc: str | None = None
    limitations: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "schema_version": self.schema_version,
            "before": {"analysis_id": self.before_analysis_id},
            "after": {"analysis_id": self.after_analysis_id},
            "compatibility": self.compatibility.to_dict(),
            "regions": {
                "matched": [dict(m) for m in self.matched],
                "added": [dict(a) for a in self.added],
                "removed": [dict(r) for r in self.removed],
            },
            "limitations": list(self.limitations),
        }
        if self.created_at_utc is not None:
            obj["created_at_utc"] = self.created_at_utc
        obj.update(self.extras)
        return obj

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != COMPARISON_SCHEMA_VERSION:
            errors.append(f"schema_version 必须是 {COMPARISON_SCHEMA_VERSION!r}")
        if not self.before_analysis_id or not self.after_analysis_id:
            errors.append("before/after analysis_id 必须非空")
        for row in self.matched:
            if "delta_pp" not in row:
                errors.append(f"matched 行缺少 delta_pp：{row.get('id')!r}")
        if errors:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "comparison 记录校验失败", {"errors": errors})


def build_comparison(
    before: AnalysisRecord, after: AnalysisRecord, created_at_utc: str | None = None
) -> ComparisonRecord:
    """兼容性校验 + 稳定 ID 配对 + delta_pp + 新增/移除单列。

    不兼容时抛 COMPARISON_INCOMPATIBLE（退出码 6），details 保留全部原因。
    """
    report = check_compatibility(before, after)
    if not report.compatible:
        raise UiAttentionError(
            ErrorCode.COMPARISON_INCOMPATIBLE,
            "A/B 配置不兼容，拒绝正式比较（原因见 details.mismatches；分表并列时须显式标注差异项）",
            {"mismatches": [dict(m) for m in report.mismatches], "canvas_note": report.canvas_note},
        )
    pairing = compare_region_rows(before.regions, after.regions)
    limitations = [
        "delta_pp 只说明模型预测分布发生变化，不表示点击率或任务效率因而提高（validation-plan §5）",
        "新增/移除区域未编造缺失一侧的 0 值，不参与差值计算",
    ]
    if report.exploratory:
        limitations.append("画面状态/任务不同：本比较为探索性，不作修改的因果结论")
    record = ComparisonRecord(
        before_analysis_id=before.analysis_id,
        after_analysis_id=after.analysis_id,
        compatibility=report,
        matched=tuple(pairing["matched"]),
        added=tuple(pairing["added"]),
        removed=tuple(pairing["removed"]),
        created_at_utc=created_at_utc,
        limitations=tuple(limitations),
    )
    record.validate()
    return record
