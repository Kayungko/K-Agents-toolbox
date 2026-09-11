"""AOI 几何校验数学（矩形半开边界、多边形顶点规则、自交检测）。

口径（data-contract.md §2、technical-design.md §6）：
- 像素坐标基于处理方向后的原图，原点左上角；
- 矩形范围 ``[x, x + width) × [y, y + height)``（半开边界）；
- 多边形 ``points: [[x, y], ...]``，至少三个不共线顶点，禁止自交；
- 坐标必须在原图内；零面积按输入错误处理。

本模块是 contracts/（schema 校验）与 aoi/（掩码生成）共用的唯一几何判定与
geometry 数据类来源（避免 contracts ↔ aoi 循环导入）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Point = tuple[int, int]


@dataclass(frozen=True)
class RectGeometry:
    """矩形：半开边界 ``[x, x+width) × [y, y+height)``。"""

    x: int
    y: int
    width: int
    height: int
    type: str = "rect"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "rect", "x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class PolygonGeometry:
    """多边形：``points: [[x, y], ...]``，至少三个不共线顶点，禁止自交。"""

    points: tuple[Point, ...]
    type: str = "polygon"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polygon", "points": [[x, y] for x, y in self.points]}


Geometry = RectGeometry | PolygonGeometry


def is_int(value: object) -> bool:
    """JSON 数值必须是真正的整数（拒绝 bool 与带小数 float）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def shoelace_area(points: Sequence[Point]) -> float:
    """多边形有向面积（鞋带公式）；绝对值为面积，0 表示全部顶点共线或退化。"""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _cross(ox: int, oy: int, ax: int, ay: int, bx: int, by: int) -> int:
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """已知 p、q、r 共线，判断 r 是否落在线段 pq 上（含端点）。"""
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """线段 p1p2 与 p3p4 是否相交（含端点接触与共线重叠）。"""
    d1 = _cross(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1])
    d2 = _cross(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1])
    d3 = _cross(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    d4 = _cross(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def polygon_self_intersects(points: Sequence[Point]) -> bool:
    """任意两条非相邻边相交（含接触）即判定自交。

    相邻边共享端点属正常；首尾边（0 与 n-1）也视为相邻。
    """
    n = len(points)
    if n < 4:
        return False  # 三角形不可能自交（重复顶点由 validate_polygon_points 单独拒绝）
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            if adjacent:
                continue
            if _segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True
    return False


def validate_rect(x: int, y: int, width: int, height: int) -> list[str]:
    """矩形结构校验（不含图像边界，边界用 rect_in_bounds）。返回错误消息列表。"""
    errors: list[str] = []
    for name, value in (("x", x), ("y", y), ("width", width), ("height", height)):
        if not is_int(value):
            errors.append(f"rect.{name} 必须是整数，得到 {value!r}")
    if is_int(width) and width <= 0:
        errors.append(f"rect.width 必须为正（零面积被拒绝），得到 {width}")
    if is_int(height) and height <= 0:
        errors.append(f"rect.height 必须为正（零面积被拒绝），得到 {height}")
    if is_int(x) and x < 0:
        errors.append(f"rect.x 不得为负，得到 {x}")
    if is_int(y) and y < 0:
        errors.append(f"rect.y 不得为负，得到 {y}")
    return errors


def rect_in_bounds(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> list[str]:
    """矩形半开区间 [x, x+w) × [y, y+h) 必须完整落在原图内。"""
    errors: list[str] = []
    if x + width > image_width:
        errors.append(f"rect 越界：x+width={x + width} 超出图像宽度 {image_width}")
    if y + height > image_height:
        errors.append(f"rect 越界：y+height={y + height} 超出图像高度 {image_height}")
    return errors


def _all_collinear(points: Sequence[Point]) -> bool:
    """全部顶点是否落在同一条直线上。"""
    (x0, y0), (x1, y1) = points[0], points[1]
    return all(_cross(x0, y0, x1, y1, x, y) == 0 for x, y in points[2:])


def validate_polygon_points(points: Sequence[Point]) -> list[str]:
    """多边形结构校验：≥3 顶点、整数坐标、无重复顶点、不共线、不自交。"""
    errors: list[str] = []
    n = len(points)
    if n < 3:
        errors.append(f"polygon 至少需要 3 个顶点，得到 {n}")
        return errors
    for i, pt in enumerate(points):
        if not (isinstance(pt, (tuple, list)) and len(pt) == 2 and is_int(pt[0]) and is_int(pt[1])):
            errors.append(f"polygon.points[{i}] 必须是整数 [x, y]，得到 {pt!r}")
    if errors:
        return errors
    pts = [(int(p[0]), int(p[1])) for p in points]
    if len(set(pts)) != n:
        errors.append("polygon 存在重复顶点")
    if shoelace_area(pts) == 0:
        # 面积=0 有两种成因：真共线（拒绝），或蝴蝶结自交导致有向面积抵消（按自交拒绝）。
        if _all_collinear(pts):
            errors.append("polygon 顶点全部共线或退化（面积=0，拒绝）")
        elif polygon_self_intersects(pts):
            errors.append("polygon 自交（非相邻边相交或接触，拒绝）")
        else:
            errors.append("polygon 退化（面积=0，拒绝）")
        return errors
    if polygon_self_intersects(pts):
        errors.append("polygon 自交（非相邻边相交或接触，拒绝）")
    return errors


def polygon_in_bounds(points: Sequence[Point], image_width: int, image_height: int) -> list[str]:
    """多边形全部顶点必须落在原图像素范围内（0 ≤ x ≤ W-1，0 ≤ y ≤ H-1）。"""
    errors: list[str] = []
    for i, (x, y) in enumerate(points):
        if not (0 <= x < image_width and 0 <= y < image_height):
            errors.append(f"polygon.points[{i}]=({x},{y}) 越界（图像 {image_width}×{image_height}）")
    return errors


def polygon_pixel_area_upper_bound(points: Sequence[Point]) -> float:
    """解析面积（鞋带），供与像素掩码面积做一致性对照；像素面积以掩码计数为准。"""
    return abs(shoelace_area(points))
