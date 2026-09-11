"""review.json schema 校验（data-contract.md §4、technical-design.md §8）。

语义评审由 Agent 生成写入运行目录；**CLI 只校验与呈现，不生成数值**。
每条 finding 必含：``id``、``region_ids``、``evidence_refs``、``evidence_type``、
``observation``、``inference``、``recommendation``、``validation_needed``。
``review.md`` 由该内容呈现，不能额外添加未记录的数值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ErrorCode, UiAttentionError

REVIEW_SCHEMA_VERSION = "game-ui-attention-review/v1"

#: technical-design §8：证据类型四分类
EVIDENCE_TYPES: tuple[str, ...] = ("computed", "observed", "inferred", "unverified")

_REVIEW_KEYS = {"schema_version", "analysis_id", "findings", "created_at_utc", "author"}
_FINDING_KEYS = {
    "id",
    "region_ids",
    "evidence_refs",
    "evidence_type",
    "observation",
    "inference",
    "recommendation",
    "validation_needed",
}


@dataclass(frozen=True)
class ReviewRecord:
    """review.json 的内存形态（校验通过后）。"""

    findings: tuple[dict[str, Any], ...]
    analysis_id: str | None = None
    schema_version: str = REVIEW_SCHEMA_VERSION
    created_at_utc: str | None = None
    author: str | None = None

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "schema_version": self.schema_version,
            "findings": [dict(f) for f in self.findings],
        }
        if self.analysis_id is not None:
            obj["analysis_id"] = self.analysis_id
        if self.created_at_utc is not None:
            obj["created_at_utc"] = self.created_at_utc
        if self.author is not None:
            obj["author"] = self.author
        return obj

    @classmethod
    def from_dict(cls, obj: Any) -> ReviewRecord:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "review 必须是 JSON 对象")
        version = obj.get("schema_version")
        if version != REVIEW_SCHEMA_VERSION:
            raise UiAttentionError(
                ErrorCode.INVALID_SCHEMA_VERSION,
                f"未知 review schema_version {version!r}（本实现只接受 {REVIEW_SCHEMA_VERSION!r}）",
                {"got": version, "expected": REVIEW_SCHEMA_VERSION},
            )
        unknown = set(obj) - _REVIEW_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "review 含未知字段", {"unknown": sorted(unknown)})
        raw_findings = obj.get("findings")
        if not isinstance(raw_findings, list):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "review.findings 必须是列表")
        errors: list[str] = []
        seen_ids: set[str] = set()
        findings: list[dict[str, Any]] = []
        for i, f in enumerate(raw_findings):
            where = f"findings[{i}]"
            if not isinstance(f, dict):
                errors.append(f"{where} 必须是对象")
                continue
            extra = set(f) - _FINDING_KEYS
            if extra:
                errors.append(f"{where} 含未知字段 {sorted(extra)}")
            fid = f.get("id")
            if not (isinstance(fid, str) and fid):
                errors.append(f"{where}.id 必须是非空字符串")
            elif fid in seen_ids:
                errors.append(f"{where}.id 重复：{fid!r}")
            else:
                seen_ids.add(fid)
            for key in ("region_ids", "evidence_refs"):
                value = f.get(key)
                if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                    errors.append(f"{where}.{key} 必须是 list[str]（证据引用不得为空想数值）")
            if f.get("evidence_type") not in EVIDENCE_TYPES:
                errors.append(f"{where}.evidence_type 必须是 {EVIDENCE_TYPES} 之一，得到 {f.get('evidence_type')!r}")
            for key in ("observation", "inference", "recommendation"):
                if not isinstance(f.get(key), str):
                    errors.append(f"{where}.{key} 必须是字符串")
            if not isinstance(f.get("validation_needed"), (str, bool)):
                errors.append(f"{where}.validation_needed 必须是字符串或布尔")
            findings.append(dict(f))
        if errors:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "review finding 结构校验失败", {"errors": errors})
        return cls(
            findings=tuple(findings),
            analysis_id=obj.get("analysis_id"),
            schema_version=version,
            created_at_utc=obj.get("created_at_utc"),
            author=obj.get("author"),
        )


def load_review(path: str | Path) -> ReviewRecord:
    """读取并校验 review.json（CLI 只校验与呈现，不生成数值）。"""
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UiAttentionError(ErrorCode.INVALID_REQUEST, f"review 文件不存在：{p.name}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise UiAttentionError(
            ErrorCode.INVALID_REQUEST, f"review 文件读取失败：{p.name}", {"reason": str(exc)}
        ) from exc
    return ReviewRecord.from_dict(obj)
