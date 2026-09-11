"""独立 regions 文件 schema（summarize 输入 / report 导出）。

data-contract §2：修改标注的独立 regions 文件**必须携带对应图片 SHA-256**，
防止应用到其他图片；哈希不匹配时拒绝（REGIONS_IMAGE_MISMATCH，退出码 2）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ErrorCode, UiAttentionError
from .request import RegionSpec

REGIONS_SCHEMA_VERSION = "game-ui-attention-regions/v1"

_REGIONS_FILE_KEYS = {"schema_version", "image_sha256", "regions"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RegionsFile:
    """独立区域文件：schema_version + image_sha256 + regions。"""

    image_sha256: str
    regions: tuple[RegionSpec, ...]
    schema_version: str = REGIONS_SCHEMA_VERSION

    def check_ids_unique(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for r in self.regions:
            if r.id in seen:
                duplicates.append(r.id)
            seen.add(r.id)
        if duplicates:
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                "regions 文件存在重复 ID（拒绝）",
                {"duplicate_ids": sorted(set(duplicates))},
            )

    def check_image_sha256(self, actual_sha256: str) -> None:
        """图片哈希改变时必须重新检查区域，不沿用过期位置（technical-design §4.6）。"""
        if self.image_sha256 != actual_sha256:
            raise UiAttentionError(
                ErrorCode.REGIONS_IMAGE_MISMATCH,
                "regions 文件的 image_sha256 与目标图片不符，拒绝应用（防止标注落到其他图片）",
                {"expected": self.image_sha256, "actual": actual_sha256},
            )

    def check_bounds(self, image_width: int, image_height: int) -> None:
        for r in self.regions:
            r.check_bounds(image_width, image_height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_sha256": self.image_sha256,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, obj: Any) -> RegionsFile:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "regions 文件必须是 JSON 对象")
        version = obj.get("schema_version")
        if version != REGIONS_SCHEMA_VERSION:
            raise UiAttentionError(
                ErrorCode.INVALID_SCHEMA_VERSION,
                f"未知 schema_version {version!r}（本实现只接受 {REGIONS_SCHEMA_VERSION!r}）",
                {"got": version, "expected": REGIONS_SCHEMA_VERSION},
            )
        unknown = set(obj) - _REGIONS_FILE_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "regions 文件含未知字段", {"unknown": sorted(unknown)})
        sha = obj.get("image_sha256")
        if not (isinstance(sha, str) and _SHA256_PATTERN.match(sha)):
            # 独立 regions 文件必须携带图片 SHA-256（缺失同样拒绝）。
            raise UiAttentionError(
                ErrorCode.INVALID_REQUEST,
                "regions 文件必须携带对应图片的 SHA-256（image_sha256，64 位小写十六进制）",
                {"got": sha},
            )
        raw_regions = obj.get("regions")
        if not isinstance(raw_regions, list):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "regions 必须是列表")
        regions = tuple(RegionSpec.from_dict(r, f"regions[{i}]") for i, r in enumerate(raw_regions))
        rf = cls(image_sha256=sha, regions=regions)
        rf.check_ids_unique()
        return rf


def load_regions_file(path: str | Path) -> RegionsFile:
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UiAttentionError(ErrorCode.INVALID_REQUEST, f"regions 文件不存在：{p.name}") from exc
    except json.JSONDecodeError as exc:
        raise UiAttentionError(
            ErrorCode.INVALID_REQUEST, f"regions 文件不是合法 JSON：{p.name}", {"reason": str(exc)}
        ) from exc
    except OSError as exc:
        raise UiAttentionError(ErrorCode.IO_ERROR, f"regions 文件读取失败：{p.name}", {"reason": str(exc)}) from exc
    return RegionsFile.from_dict(obj)
