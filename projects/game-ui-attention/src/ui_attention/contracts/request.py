"""AnalyzeRequest v1 schema 与校验（data-contract.md §2）。

拒绝项（§2 + 任务口径）：未知 schema_version、重复 ID、零面积、自交多边形、越界坐标、
未知字段、非法枚举值。相对图片路径相对于 request 文件目录解析。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..aoi.geometry import (
    Geometry,
    PolygonGeometry,
    RectGeometry,
    polygon_in_bounds,
    rect_in_bounds,
    validate_polygon_points,
    validate_rect,
)
from ..errors import ErrorCode, UiAttentionError

REQUEST_SCHEMA_VERSION = "game-ui-attention-request/v1"

REGION_SOURCES: tuple[str, ...] = ("manual", "agent", "imported")
REGION_STATUSES: tuple[str, ...] = ("candidate", "confirmed")

_REQUEST_KEYS = {"schema_version", "image", "player_goal", "screen_type", "backend_profile", "regions"}
_REGION_KEYS = {"id", "label", "role", "geometry", "source", "status", "source_reference"}
_RECT_KEYS = {"type", "x", "y", "width", "height"}
_POLYGON_KEYS = {"type", "points"}


def parse_geometry(obj: Any, where: str) -> Geometry:
    """解析并结构校验 geometry（不含图像边界；边界用 RegionSpec.check_bounds）。"""
    if not isinstance(obj, dict):
        raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.geometry 必须是对象", {"got": type(obj).__name__})
    gtype = obj.get("type")
    if gtype == "rect":
        unknown = set(obj) - _RECT_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.geometry 含未知字段", {"unknown": sorted(unknown)})
        errors = validate_rect(obj.get("x"), obj.get("y"), obj.get("width"), obj.get("height"))
        if errors:
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.geometry 无效", {"errors": errors})
        return RectGeometry(x=obj["x"], y=obj["y"], width=obj["width"], height=obj["height"])
    if gtype == "polygon":
        unknown = set(obj) - _POLYGON_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.geometry 含未知字段", {"unknown": sorted(unknown)})
        points = obj.get("points")
        if not isinstance(points, list):
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.geometry.points 必须是列表")
        pts: list[tuple[int, int]] = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                pts.append((pt[0], pt[1]))
            else:
                pts.append(pt)  # 交由 validate 报错
        errors = validate_polygon_points(pts)
        if errors:
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.geometry 无效", {"errors": errors})
        return PolygonGeometry(points=tuple(pts))  # type: ignore[arg-type]
    raise UiAttentionError(
        ErrorCode.INVALID_AOI,
        f"{where}.geometry.type 必须是 rect|polygon，得到 {gtype!r}",
    )


@dataclass(frozen=True)
class RegionSpec:
    """AOI 区域（data-contract §2）。

    ``confirmed`` 表示边界经确认，不表示模型或设计效果通过验证。
    导入设计节点（source=imported）时 ``source_reference`` 附来源引用与实际坐标变换记录；
    不得仅凭节点名称关联截图。
    """

    id: str
    geometry: Geometry
    source: str
    status: str
    label: str | None = None
    role: str | None = None
    source_reference: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "role": self.role,
            "geometry": self.geometry.to_dict(),
            "source": self.source,
            "status": self.status,
        }
        if self.source_reference is not None:
            obj["source_reference"] = self.source_reference
        return obj

    def check_bounds(self, image_width: int, image_height: int) -> None:
        """坐标必须在（方向处理后的）原图内；越界拒绝。"""
        if isinstance(self.geometry, RectGeometry):
            g = self.geometry
            errors = rect_in_bounds(g.x, g.y, g.width, g.height, image_width, image_height)
        else:
            errors = polygon_in_bounds(self.geometry.points, image_width, image_height)
        if errors:
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                f"区域 {self.id!r} 坐标越界",
                {"region_id": self.id, "errors": errors, "image": [image_width, image_height]},
            )

    @classmethod
    def from_dict(cls, obj: Any, where: str = "region") -> RegionSpec:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where} 必须是对象")
        unknown = set(obj) - _REGION_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where} 含未知字段", {"unknown": sorted(unknown)})
        rid = obj.get("id")
        if not (isinstance(rid, str) and rid):
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"{where}.id 必须是非空字符串", {"got": repr(rid)})
        source = obj.get("source")
        if source not in REGION_SOURCES:
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                f"区域 {rid!r} 的 source 必须是 {REGION_SOURCES} 之一，得到 {source!r}",
            )
        status = obj.get("status")
        if status not in REGION_STATUSES:
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                f"区域 {rid!r} 的 status 必须是 {REGION_STATUSES} 之一，得到 {status!r}"
                "（confirmed 仅表示边界经确认，不表示效果通过验证）",
            )
        label = obj.get("label")
        if label is not None and not isinstance(label, str):
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"区域 {rid!r} 的 label 必须是字符串|None")
        role = obj.get("role")
        if role is not None and not isinstance(role, str):
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"区域 {rid!r} 的 role 必须是字符串|None")
        source_ref = obj.get("source_reference")
        if source_ref is not None and not isinstance(source_ref, dict):
            raise UiAttentionError(ErrorCode.INVALID_AOI, f"区域 {rid!r} 的 source_reference 必须是对象|None")
        if source == "imported" and not source_ref:
            # 导入设计节点必须附 source reference 和实际坐标变换记录；不得仅凭节点名称关联截图。
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                f"区域 {rid!r} source=imported 时 source_reference 必须附来源引用与坐标变换记录，不得缺失或为空",
            )
        geometry = parse_geometry(obj.get("geometry"), f"{where}({rid})")
        return cls(
            id=rid, geometry=geometry, source=source, status=status, label=label, role=role, source_reference=source_ref
        )


@dataclass(frozen=True)
class AnalyzeRequest:
    """AnalyzeRequest v1（data-contract §2）。

    ``player_goal`` 和 ``regions`` 可缺省；缺少目标时下游不得声称目标完成路径合理。
    ``backend_profile`` 必须是已登记、不可在执行时静默改写的配置。
    """

    image: str
    backend_profile: str
    player_goal: str | None = None
    screen_type: str | None = None
    regions: tuple[RegionSpec, ...] = ()
    schema_version: str = REQUEST_SCHEMA_VERSION
    base_dir: Path | None = field(default=None, compare=False, repr=False)

    @property
    def image_path(self) -> Path:
        """相对图片路径相对于 request 文件目录解析，不相对于 shell 当前目录。"""
        p = Path(self.image)
        if p.is_absolute() or self.base_dir is None:
            return p
        return self.base_dir / p

    def check_region_ids_unique(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for r in self.regions:
            if r.id in seen:
                duplicates.append(r.id)
            seen.add(r.id)
        if duplicates:
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                "regions 存在重复 ID（拒绝）",
                {"duplicate_ids": sorted(set(duplicates))},
            )

    def check_bounds(self, image_width: int, image_height: int) -> None:
        for r in self.regions:
            r.check_bounds(image_width, image_height)

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "schema_version": self.schema_version,
            "image": self.image,
            "backend_profile": self.backend_profile,
        }
        if self.player_goal is not None:
            obj["player_goal"] = self.player_goal
        if self.screen_type is not None:
            obj["screen_type"] = self.screen_type
        if self.regions:
            obj["regions"] = [r.to_dict() for r in self.regions]
        return obj

    @classmethod
    def from_dict(cls, obj: Any, base_dir: Path | None = None) -> AnalyzeRequest:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "AnalyzeRequest 必须是 JSON 对象")
        version = obj.get("schema_version")
        if version != REQUEST_SCHEMA_VERSION:
            # 拒绝未知 schema 版本：不猜测迁移、不按最新版本解释。
            raise UiAttentionError(
                ErrorCode.INVALID_SCHEMA_VERSION,
                f"未知 schema_version {version!r}（本实现只接受 {REQUEST_SCHEMA_VERSION!r}）",
                {"got": version, "expected": REQUEST_SCHEMA_VERSION},
            )
        unknown = set(obj) - _REQUEST_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "AnalyzeRequest 含未知字段", {"unknown": sorted(unknown)})
        image = obj.get("image")
        if not (isinstance(image, str) and image):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "image 必须是非空字符串路径")
        profile = obj.get("backend_profile")
        if not (isinstance(profile, str) and profile):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "backend_profile 必须是非空字符串（已登记配置别名）")
        goal = obj.get("player_goal")
        if goal is not None and not isinstance(goal, str):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "player_goal 必须是字符串|None")
        screen_type = obj.get("screen_type")
        if screen_type is not None and not isinstance(screen_type, str):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "screen_type 必须是字符串|None")
        raw_regions = obj.get("regions", [])
        if not isinstance(raw_regions, list):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "regions 必须是列表|缺省")
        regions = tuple(RegionSpec.from_dict(r, f"regions[{i}]") for i, r in enumerate(raw_regions))
        req = cls(
            image=image,
            backend_profile=profile,
            player_goal=goal,
            screen_type=screen_type,
            regions=regions,
            base_dir=base_dir,
        )
        req.check_region_ids_unique()
        return req


def load_analyze_request(path: str | Path) -> AnalyzeRequest:
    """读取 request.json 并校验；base_dir 记录 request 文件所在目录。"""
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UiAttentionError(ErrorCode.INVALID_REQUEST, f"request 文件不存在：{p.name}") from exc
    except json.JSONDecodeError as exc:
        raise UiAttentionError(
            ErrorCode.INVALID_REQUEST, f"request 文件不是合法 JSON：{p.name}", {"reason": str(exc)}
        ) from exc
    except OSError as exc:
        raise UiAttentionError(ErrorCode.IO_ERROR, f"request 文件读取失败：{p.name}", {"reason": str(exc)}) from exc
    return AnalyzeRequest.from_dict(obj, base_dir=p.parent)
