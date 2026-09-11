"""regions JSON 导出与校验（纯函数，供 C1 cli 接线）。

schema 以 C1 冻结契约为准（contracts/regions.py、contracts/request.py、
aoi/geometry.py，2026-09-11 集成信号）：

- ``REGIONS_SCHEMA_VERSION = "game-ui-attention-regions/v1"``（C1 冻结）。
- 区域条目 = ``RegionSpec``：``id`` / ``label`` (str|None) / ``role`` (str|None) /
  ``geometry`` / ``source ∈ {manual, agent, imported}`` /
  ``status ∈ {candidate, confirmed}`` / 可选 ``source_reference``
  （``source == "imported"`` 时必须非空——不得仅凭节点名称关联截图）。
- 矩形：``[x, x+width) × [y, y+height)`` 半开边界；``x, y >= 0``、
  ``width, height >= 1``、``x+width <= W``、``y+height <= H``；整数坐标。
- 多边形：``points: [[x, y], ...]`` ≥3 个整数顶点、**无重复顶点**
  （含首尾闭合写法，C1 口径拒绝）、不共线（鞋带面积 ≠ 0）、不自交
  （非相邻边相交或接触即自交，aoi.geometry.polygon_self_intersects 口径）、
  顶点严格落在像素范围内 ``0 <= x < W``、``0 <= y < H``
  （aoi.geometry.polygon_in_bounds 口径）。
- **独立 regions 文件必须携带图片 SHA-256**（data-contract §2）；
  ``expected_image_sha256`` 给定时不匹配即拒绝（防止标注应用到其他图片，
  对应 C1 ErrorCode.REGIONS_IMAGE_MISMATCH，退出码 2）。

双层校验设计：

1. 细粒度错误收集（本模块 collector）：一次调用列出全部问题，错误码面向
   HTML 圈选导入的展示反馈（ZERO_AREA_REGION / SELF_INTERSECTING_POLYGON /
   REGION_OUT_OF_BOUNDS / NON_INTEGER_COORD / DUPLICATE_REGION_ID …）。
2. C1 权威兜底：细粒度检查全部通过后，把 payload 交给冻结契约
   ``RegionsFile.from_dict`` + ``check_bounds`` + ``check_image_sha256``
   复核；任何不一致以 ``CONTRACT_REJECTED`` 报出（C1 契约是唯一权威，
   本层保证"通过我校验的必通过 C1 校验"）。

主要公开接口（纯函数式）：

- ``compute_image_sha256(path) -> str``
- ``normalize_region_for_export(region) -> dict``（对齐 RegionSpec.to_dict 形态）
- ``make_regions_payload(regions, image_sha256, schema_version=REGIONS_SCHEMA_VERSION) -> dict``
- ``export_regions_file(*, regions, image_sha256, out_path, schema_version=...) -> Path``
- ``validate_regions_payload(payload, *, expected_image_sha256=None, image_size=None) -> RegionsValidationResult``
- ``load_and_validate_regions_file(path, *, expected_image_sha256=None, image_size=None)
  -> tuple[dict | None, RegionsValidationResult]``

导出 JSON 为确定性文本（indent=2、UTF-8、LF 换行、末尾单换行），
同输入字节级一致。HTML 圈选结果导入 = 浏览器下载的 JSON 文本经
``validate_regions_payload`` 全量校验后由 C1 进入 summarize 流程；
本模块不隐式写回任何文件。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.regions import REGIONS_SCHEMA_VERSION, RegionsFile
from ..contracts.request import REGION_SOURCES, REGION_STATUSES
from ..errors import UiAttentionError

__all__ = [
    "GEOMETRY_TYPES",
    "REGIONS_SCHEMA_V1",
    "REGIONS_SCHEMA_VERSION",
    "REGIONS_SCHEMA_VERSIONS",
    "REGION_SOURCES",
    "REGION_STATUSES",
    "RegionsValidationResult",
    "compute_image_sha256",
    "export_regions_file",
    "load_and_validate_regions_file",
    "make_regions_payload",
    "normalize_region_for_export",
    "validate_regions_payload",
]

# C1 冻结版本号为唯一权威；别名与元组形式仅为兼容既有引用。
REGIONS_SCHEMA_V1 = REGIONS_SCHEMA_VERSION
REGIONS_SCHEMA_VERSIONS: tuple[str, ...] = (REGIONS_SCHEMA_VERSION,)
GEOMETRY_TYPES: tuple[str, ...] = ("rect", "polygon")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGION_KEYS = {"id", "label", "role", "geometry", "source", "status", "source_reference"}


@dataclass(frozen=True)
class RegionsValidationResult:
    """校验结果：``ok`` 为 True 当且仅当 ``errors`` 为空。"""

    ok: bool
    errors: tuple[dict[str, Any], ...] = ()

    def error_codes(self) -> tuple[str, ...]:
        return tuple(e["code"] for e in self.errors)


@dataclass
class _Collector:
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, message: str, region_id: Any = None) -> None:
        entry: dict[str, Any] = {"code": code, "message": message}
        if region_id is not None:
            entry["region_id"] = region_id
        self.errors.append(entry)

    def result(self) -> RegionsValidationResult:
        return RegionsValidationResult(ok=not self.errors, errors=tuple(self.errors))


# --------------------------------------------------------------------- hash
def compute_image_sha256(path: Path | str) -> str:
    """计算图片文件 SHA-256（小写 hex），用于标注绑定。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- export
def normalize_region_for_export(region: dict[str, Any]) -> dict[str, Any]:
    """把区域 dict 规整为 ``RegionSpec.to_dict()`` 形态（导出/HTML 下载用）。

    - 键序固定：id, label, role, geometry, source, status[, source_reference]。
    - ``label`` / ``role`` 恒存在：缺失或空串规整为 ``None``（JSON null），
      与 C1 契约（str|None）一致，保证导出 → C1 from_dict 往返可解析。
    - ``geometry`` / ``source_reference`` 原样深拷贝（结构不变）。
    """
    if not isinstance(region, dict):
        raise ValueError(f"region must be a dict, got {type(region)}")
    label = region.get("label")
    if label is not None and (not isinstance(label, str) or label == ""):
        label = None
    role = region.get("role")
    if role is not None and (not isinstance(role, str) or role == ""):
        role = None
    out: dict[str, Any] = {
        "id": region.get("id"),
        "label": label,
        "role": role,
        "geometry": json.loads(json.dumps(region.get("geometry"))),
        "source": region.get("source"),
        "status": region.get("status"),
    }
    if region.get("source_reference") is not None:
        out["source_reference"] = json.loads(json.dumps(region["source_reference"]))
    return out


def make_regions_payload(
    regions: list[dict[str, Any]],
    image_sha256: str,
    schema_version: str = REGIONS_SCHEMA_VERSION,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    """构造独立 regions 文件 payload（不落盘）。

    强制携带图片 SHA-256（data-contract §2）。``normalize=True`` 时区域条目
    经 :func:`normalize_region_for_export` 规整为 RegionSpec.to_dict 形态；
    ``False`` 时原样深拷贝（结构不变，往返一致由测试保证）。
    """
    if not isinstance(image_sha256, str) or not _SHA256_RE.match(image_sha256):
        raise ValueError(
            "image_sha256 must be a 64-char lowercase hex string "
            "(standalone regions file must carry the image SHA-256)"
        )
    if schema_version not in REGIONS_SCHEMA_VERSIONS:
        raise ValueError(
            f"unknown schema_version {schema_version!r}; known: {REGIONS_SCHEMA_VERSIONS}"
        )
    if not isinstance(regions, list):
        raise ValueError(f"regions must be a list, got {type(regions)}")
    if normalize:
        body = [normalize_region_for_export(r) for r in regions]
    else:
        body = json.loads(json.dumps(regions))  # 确定性深拷贝
    return {
        "schema_version": schema_version,
        "image_sha256": image_sha256,
        "regions": body,
    }


def export_regions_file(
    *,
    regions: list[dict[str, Any]],
    image_sha256: str,
    out_path: Path | str,
    schema_version: str = REGIONS_SCHEMA_VERSION,
    normalize: bool = True,
) -> Path:
    """导出 regions JSON 文件；返回输出路径。

    确定性文本：键序按 payload 构造顺序，indent=2，UTF-8，LF 换行，
    末尾单个换行符。同输入字节级一致。
    """
    payload = make_regions_payload(
        regions, image_sha256, schema_version, normalize=normalize
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


# --------------------------------------------------------------- geometry
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _orient(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def _on_segment(ax, ay, bx, by, cx, cy) -> bool:
    """共线前提下，点 C 是否落在线段 AB 上（含端点）。"""
    return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """线段 p1p2 与 p3p4 是否相交（含端点接触/共线重叠）。

    与 C1 aoi.geometry._segments_intersect 同口径。
    """
    o1 = _orient(*p1, *p2, *p3)
    o2 = _orient(*p1, *p2, *p4)
    o3 = _orient(*p3, *p4, *p1)
    o4 = _orient(*p3, *p4, *p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(*p1, *p2, *p3):
        return True
    if o2 == 0 and _on_segment(*p1, *p2, *p4):
        return True
    if o3 == 0 and _on_segment(*p3, *p4, *p1):
        return True
    if o4 == 0 and _on_segment(*p3, *p4, *p2):
        return True
    return False


def _polygon_signed_area2(points: list[tuple[int, int]]) -> int:
    """鞋带公式 ×2（整数精确）。"""
    s = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def _all_collinear(points: list[tuple[int, int]]) -> bool:
    (x0, y0), (x1, y1) = points[0], points[1]
    return all(_orient(x0, y0, x1, y1, x, y) == 0 for x, y in points[2:])


def _polygon_self_intersects(points: list[tuple[int, int]]) -> bool:
    """任意两条非相邻边相交（含接触）即自交；三角形不可能自交（C1 口径）。"""
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            if adjacent:
                continue
            if _segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                return True
    return False


def _validate_geometry(
    geom: Any,
    image_size: tuple[int, int] | None,
    rid: Any,
    c: _Collector,
) -> None:
    if not isinstance(geom, dict):
        c.add("BAD_GEOMETRY", "geometry must be an object", rid)
        return
    gtype = geom.get("type")
    if gtype not in GEOMETRY_TYPES:
        c.add(
            "BAD_GEOMETRY",
            f"geometry.type must be one of {GEOMETRY_TYPES}, got {gtype!r}",
            rid,
        )
        return
    if gtype == "rect":
        _validate_rect(geom, image_size, rid, c)
    else:
        _validate_polygon(geom, image_size, rid, c)


def _validate_rect(
    geom: dict[str, Any],
    image_size: tuple[int, int] | None,
    rid: Any,
    c: _Collector,
) -> None:
    unknown = set(geom) - {"type", "x", "y", "width", "height"}
    if unknown:
        c.add("BAD_GEOMETRY", f"rect geometry has unknown keys: {sorted(unknown)}", rid)
        return
    coords = {k: geom.get(k) for k in ("x", "y", "width", "height")}
    for k, v in coords.items():
        if not _is_int(v):
            c.add(
                "NON_INTEGER_COORD",
                f"rect {k} must be an int (HTML export rounds to int), got {v!r}",
                rid,
            )
            return
    x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
    if w <= 0 or h <= 0:
        c.add("ZERO_AREA_REGION", f"rect has non-positive area (w={w}, h={h})", rid)
        return
    if x < 0 or y < 0:
        c.add("REGION_OUT_OF_BOUNDS", f"rect origin out of bounds (x={x}, y={y})", rid)
        return
    if image_size is not None:
        iw, ih = image_size
        if x + w > iw or y + h > ih:
            c.add(
                "REGION_OUT_OF_BOUNDS",
                f"rect exceeds image bounds: x+w={x + w} > {iw} or y+h={y + h} > {ih}",
                rid,
            )


def _validate_polygon(
    geom: dict[str, Any],
    image_size: tuple[int, int] | None,
    rid: Any,
    c: _Collector,
) -> None:
    unknown = set(geom) - {"type", "points"}
    if unknown:
        c.add("BAD_GEOMETRY", f"polygon geometry has unknown keys: {sorted(unknown)}", rid)
        return
    pts = geom.get("points")
    if not isinstance(pts, list):
        c.add("BAD_GEOMETRY", "polygon points must be a list of [x, y]", rid)
        return
    if len(pts) < 3:
        c.add("BAD_GEOMETRY", f"polygon needs >= 3 vertices, got {len(pts)}", rid)
        return
    norm: list[tuple[int, int]] = []
    for p in pts:
        if (
            not isinstance(p, (list, tuple))
            or len(p) != 2
            or not _is_int(p[0])
            or not _is_int(p[1])
        ):
            c.add("NON_INTEGER_COORD", f"polygon vertex must be [int, int], got {p!r}", rid)
            return
        norm.append((int(p[0]), int(p[1])))
    # 重复顶点拒绝（C1 口径：含首尾闭合写法，不做自动去闭合）
    if len(set(norm)) != len(norm):
        c.add(
            "BAD_GEOMETRY",
            "polygon has duplicate vertices (closing point equal to first is rejected; "
            "point list must not repeat vertices)",
            rid,
        )
        return
    # 越界：顶点严格落在像素范围内 0 <= x < W、0 <= y < H（C1 polygon_in_bounds 口径）
    if image_size is not None:
        iw, ih = image_size
        for x, y in norm:
            if not (0 <= x < iw and 0 <= y < ih):
                c.add(
                    "REGION_OUT_OF_BOUNDS",
                    f"polygon vertex ({x}, {y}) outside pixel bounds [0,{iw}) x [0,{ih})",
                    rid,
                )
                return
    # 零面积 / 共线 / 自交（面积=0 成因细分，对齐 C1 validate_polygon_points 口径）
    area2 = _polygon_signed_area2(norm)
    if area2 == 0:
        if _all_collinear(norm):
            c.add(
                "ZERO_AREA_REGION",
                "polygon vertices are all collinear or degenerate (area = 0)",
                rid,
            )
        elif _polygon_self_intersects(norm):
            c.add(
                "SELF_INTERSECTING_POLYGON",
                "polygon self-intersects (bowtie: signed area cancels to 0)",
                rid,
            )
        else:
            c.add("ZERO_AREA_REGION", "polygon degenerate (area = 0)", rid)
        return
    if _polygon_self_intersects(norm):
        c.add(
            "SELF_INTERSECTING_POLYGON",
            "polygon self-intersects (non-adjacent edges intersect or touch)",
            rid,
        )


# --------------------------------------------------------------- validate
def validate_regions_payload(
    payload: Any,
    *,
    expected_image_sha256: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> RegionsValidationResult:
    """校验独立 regions payload（含 HTML 圈选导出结果的导入校验）。

    参数：
      - ``expected_image_sha256``：给定时执行绑定校验，不匹配 →
        IMAGE_SHA256_MISMATCH（C1 权威口径 REGIONS_IMAGE_MISMATCH，退出码 2）。
      - ``image_size``：``(width, height)``；给定时执行越界校验。

    细粒度错误码：UNKNOWN_SCHEMA_VERSION、MISSING_IMAGE_SHA256、
    BAD_IMAGE_SHA256、IMAGE_SHA256_MISMATCH、BAD_REGIONS、BAD_REGION_ID、
    DUPLICATE_REGION_ID、BAD_REGION_FIELD、BAD_REGION_SOURCE、
    BAD_REGION_STATUS、BAD_GEOMETRY、NON_INTEGER_COORD、ZERO_AREA_REGION、
    REGION_OUT_OF_BOUNDS、SELF_INTERSECTING_POLYGON、CONTRACT_REJECTED。

    细粒度检查全部通过后，交由 C1 冻结契约（RegionsFile.from_dict +
    check_bounds + check_image_sha256）权威复核；复核失败以
    CONTRACT_REJECTED 报出（正常情况下不可达，可达即口径漂移，须报总控）。
    """
    c = _Collector()
    if not isinstance(payload, dict):
        c.add("BAD_REGIONS", f"payload must be an object, got {type(payload)}")
        return c.result()

    # schema 版本
    sv = payload.get("schema_version")
    if sv not in REGIONS_SCHEMA_VERSIONS:
        c.add(
            "UNKNOWN_SCHEMA_VERSION",
            f"schema_version must be one of {REGIONS_SCHEMA_VERSIONS}, got {sv!r}",
        )

    # 图片 SHA-256 绑定（独立 regions 文件必须携带）
    sha = payload.get("image_sha256")
    if sha is None:
        c.add(
            "MISSING_IMAGE_SHA256",
            "standalone regions file must carry the image SHA-256 (data-contract §2)",
        )
    elif not isinstance(sha, str) or not _SHA256_RE.match(sha):
        c.add("BAD_IMAGE_SHA256", f"image_sha256 must be 64-char lowercase hex, got {sha!r}")
    elif expected_image_sha256 is not None and sha != expected_image_sha256.lower():
        c.add(
            "IMAGE_SHA256_MISMATCH",
            "regions file is bound to a different image "
            f"(expected {expected_image_sha256}, got {sha})",
        )

    unknown_top = set(payload) - {"schema_version", "image_sha256", "regions"}
    if unknown_top:
        c.add("BAD_REGIONS", f"payload has unknown top-level keys: {sorted(unknown_top)}")

    regions = payload.get("regions")
    if not isinstance(regions, list):
        c.add("BAD_REGIONS", f"regions must be a list, got {type(regions)}")
        return c.result()

    seen_ids: set[str] = set()
    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            c.add("BAD_REGIONS", f"regions[{i}] must be an object")
            continue
        rid = region.get("id")
        if not isinstance(rid, str) or not rid.strip():
            c.add("BAD_REGION_ID", f"regions[{i}].id must be a non-empty string", rid)
            rid = None
        else:
            if rid in seen_ids:
                c.add("DUPLICATE_REGION_ID", f"duplicate region id {rid!r}", rid)
            seen_ids.add(rid)

        label = region.get("label")
        if label is not None and not isinstance(label, str):
            c.add("BAD_REGION_FIELD", f"label must be a string or null, got {label!r}", rid)
        role = region.get("role")
        if role is not None and not isinstance(role, str):
            c.add("BAD_REGION_FIELD", f"role must be a string or null, got {role!r}", rid)
        source = region.get("source")
        if source not in REGION_SOURCES:
            c.add("BAD_REGION_SOURCE", f"source must be one of {REGION_SOURCES}, got {source!r}", rid)
        status = region.get("status")
        if status not in REGION_STATUSES:
            c.add("BAD_REGION_STATUS", f"status must be one of {REGION_STATUSES}, got {status!r}", rid)
        source_ref = region.get("source_reference")
        if source_ref is not None and not isinstance(source_ref, dict):
            c.add("BAD_REGION_FIELD", f"source_reference must be an object or null, got {source_ref!r}", rid)
        if source == "imported" and not source_ref:
            c.add(
                "BAD_REGION_FIELD",
                "source=imported requires non-empty source_reference "
                "(source reference + actual coordinate transform record)",
                rid,
            )
        unknown_keys = set(region) - _REGION_KEYS
        if unknown_keys:
            c.add("BAD_REGION_FIELD", f"region has unknown keys: {sorted(unknown_keys)}", rid)
        if "geometry" not in region:
            c.add("BAD_GEOMETRY", "region has no geometry", rid)
        else:
            _validate_geometry(region["geometry"], image_size, rid, c)

    if c.errors:
        return c.result()

    # ---- C1 冻结契约权威兜底（细粒度全过后才执行，避免重复报错）----
    try:
        rf = RegionsFile.from_dict(payload)
        if image_size is not None:
            rf.check_bounds(int(image_size[0]), int(image_size[1]))
        if expected_image_sha256 is not None:
            rf.check_image_sha256(expected_image_sha256.lower())
    except UiAttentionError as exc:
        c.add(
            "CONTRACT_REJECTED",
            f"C1 contracts rejected payload: [{exc.code.value}] {exc.message}",
        )
    return c.result()


def load_and_validate_regions_file(
    path: Path | str,
    *,
    expected_image_sha256: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[dict[str, Any] | None, RegionsValidationResult]:
    """读取 JSON 文件并全量校验；返回 ``(payload 或 None, 结果)``。

    文件不存在/JSON 解析失败产出结构化错误而非异常，供 C1 cli
    直接映射退出码 2（参数、图片或 AOI 无效）。
    """
    c = _Collector()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        c.add("BAD_REGIONS", f"cannot read regions file: {exc}")
        return None, c.result()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        c.add("BAD_REGIONS", f"regions file is not valid JSON: {exc}")
        return None, c.result()
    result = validate_regions_payload(
        payload,
        expected_image_sha256=expected_image_sha256,
        image_size=image_size,
    )
    return (payload if isinstance(payload, dict) else None), result
