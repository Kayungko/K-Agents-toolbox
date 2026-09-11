"""A/B 对比视图：共用色阶渲染 + comparison 报告页。

口径（data-contract §6、technical-design §9、validation-plan §1"色阶"行）：

- **A/B 共用颜色标尺，不各自拉满**：:func:`render_shared_overlays` 先用
  ``colorscale.shared_range(density_a, density_b)`` 求联合值域，构造**同一个**
  ``ColorScale`` 实例作用于两图；色阶参数（可序列化 dict）随结果返回，
  由 C1 写入 comparison 运行目录的 manifest.json（"共享色阶参数写入
  manifest"）。
- comparison.json 结构以 C1 冻结契约为准（contracts/comparison.py
  ``ComparisonRecord.to_dict``）：``before/after.analysis_id``、
  ``compatibility{compatible, mismatches, canvas_note, exploratory,
  exploratory_reasons}``、``regions{matched, added, removed}``、
  ``limitations``；matched 行字段为 metrics.regions.compare_region_rows
  输出（id/label/mass_before/mass_after/delta_pp/area_px_before/
  area_px_after/area_px_change/relative_density_before/
  relative_density_after/geometry_before/geometry_after）。
- 新增/移除区域**单列区块**呈现，明确"不编造缺失一侧的 0 值、不参与
  差值计算"；不兼容时（compatible=false，正常应由 CLI 退出码 6 拦截）
  渲染拒绝页：保留 mismatches 原因，不展示 delta 数值表。
- exploratory=true（画面状态/任务不同）时显著标注"探索性比较，不作
  修改的因果结论"。
- delta_pp 等全部数值**原样展示**传入 comparison 内容（仅显示舍入），
  本模块不重新计算任何统计、不从 PNG 颜色反推数值。
- HTML 安全口径与 report.py 相同：自包含（CSS/JS 内联 + 图片 data URI）、
  全文本转义、CSP 内容哈希收紧、无外部脚本/样式/字体、无自动网络请求、
  产物无 ``http(s)://`` 与 ``<script src`` 字面量。

渲染入口（纯函数式：输入数据 + 配置 → 输出文件路径）：

- ``render_shared_overlays(*, base_a, density_a, base_b, density_b,
  out_dir, name="ember", normalization="linear",
  alpha=DEFAULT_OVERLAY_ALPHA) -> SharedCompareRender``
  共用色阶渲染两侧 overlay/heatmap PNG，返回路径 + 可序列化色阶参数。
- ``render_comparison_html(*, comparison, side_a, side_b,
  colorscale_params, overlay_alpha, out_path, title=None) -> Path``
  低层入口：由 comparison dict + 两侧渲染数据组装 compare 页 HTML。
- ``render_compare_report(*, out_dir, comparison, base_a, density_a,
  regions_a, base_b, density_b, regions_b, name="ember",
  normalization="linear", alpha=DEFAULT_OVERLAY_ALPHA,
  title=None) -> dict[str, Path]``
  编排入口：共用色阶渲染 + comparison 页组装，返回
  ``{"overlay_a", "heatmap_a", "overlay_b", "heatmap_b", "report"}``。
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .colorscale import ColorScale
from .overlay import (
    DEFAULT_OVERLAY_ALPHA,
    overlay_record,
    render_heatmap_png,
    render_overlay_png,
)
from .report import (
    _csp_hash,
    _data_uri,
    _load_resource,
    _regions_svg,
    _render_value,
    esc,
    format_metric,
    json_data_island,
)

__all__ = [
    "SharedCompareRender",
    "colorbar_png_bytes",
    "render_compare_report",
    "render_comparison_html",
    "render_shared_overlays",
]


def colorbar_png_bytes(colorscale: ColorScale, width: int = 512, height: int = 16) -> bytes:
    """生成色标条 PNG 字节（LUT 横向渐变，不透明），供对比页内嵌展示。

    仅展示用：色标条颜色来自与热图完全相同的 LUT，数值轴端点以文本标注
    （vmin/vmax），观看者不得从颜色反推统计值。
    """
    lut = colorscale.lut()  # (256, 4)
    idx = np.linspace(0, len(lut) - 1, width).round().astype(int)
    row = lut[idx]
    row[:, 3] = 255  # 色标条不透明（LUT alpha 斜坡只作用于热图叠加）
    bar = np.repeat(row[None, :, :], height, axis=0)
    buf = io.BytesIO()
    Image.fromarray(bar, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class SharedCompareRender:
    """共用色阶渲染结果：两侧产物路径 + 同一份可序列化色阶参数。"""

    overlay_a: Path
    heatmap_a: Path
    overlay_b: Path
    heatmap_b: Path
    colorscale_params: dict[str, Any]
    alpha: float

    def overlay_record_dict(self) -> dict[str, Any]:
        """供 manifest 记录的展示参数（两侧共用同一份）。"""
        return overlay_record(ColorScale.from_params(self.colorscale_params), self.alpha)


def render_shared_overlays(
    *,
    base_a: np.ndarray,
    density_a: np.ndarray,
    base_b: np.ndarray,
    density_b: np.ndarray,
    out_dir: Path | str,
    name: str = "ember",
    normalization: str = "linear",
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> SharedCompareRender:
    """以 A/B 共用色阶（联合值域）渲染两侧 overlay/heatmap PNG。

    同一 ``ColorScale`` 实例作用于两图——A/B 共用颜色标尺，不各自拉满。
    density 必须已是对应原图尺寸（重采样属 C1 imaging 职责）。
    """
    out = Path(out_dir)
    shared = ColorScale.shared(name, density_a, density_b, normalization=normalization)
    oa = render_overlay_png(
        base_image=base_a, density=density_a, colorscale=shared, alpha=alpha,
        out_path=out / "overlay_a.png",
    )
    ha = render_heatmap_png(density=density_a, colorscale=shared, out_path=out / "heatmap_a.png")
    ob = render_overlay_png(
        base_image=base_b, density=density_b, colorscale=shared, alpha=alpha,
        out_path=out / "overlay_b.png",
    )
    hb = render_heatmap_png(density=density_b, colorscale=shared, out_path=out / "heatmap_b.png")
    return SharedCompareRender(
        overlay_a=oa, heatmap_a=ha, overlay_b=ob, heatmap_b=hb,
        colorscale_params=shared.to_params(), alpha=alpha,
    )


# ---------------------------------------------------------------------------
# comparison.json 消费（C1 冻结结构，防御性 get）
# ---------------------------------------------------------------------------

def _compat(comparison: dict) -> dict:
    c = comparison.get("compatibility")
    return c if isinstance(c, dict) else {}


def _region_rows(comparison: dict, key: str) -> list:
    regions = comparison.get("regions")
    if isinstance(regions, dict):
        rows = regions.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _num_cell(value: Any, cls: str = "num") -> str:
    if value is None:
        return "<td>—</td>"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<td class="{cls}">{esc(format_metric(value))}</td>'
    return f"<td>{esc(value)}</td>"


def _matched_table(matched: list) -> str:
    if not matched:
        return '<p class="muted">无按稳定 ID 配对的区域（matched 为空）。</p>'
    head = (
        "<th>id</th><th>label</th>"
        "<th>mass A</th><th>mass B</th><th>delta_pp</th>"
        "<th>area_px A</th><th>area_px B</th><th>Δarea_px</th>"
        "<th>rel_density A</th><th>rel_density B</th>"
    )
    rows = []
    for m in matched:
        if not isinstance(m, dict):
            continue
        delta = m.get("delta_pp")
        delta_cls = "num"
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            if delta > 0:
                delta_cls = "num delta-up"
            elif delta < 0:
                delta_cls = "num delta-down"
        rows.append(
            "<tr>"
            f"<td>{esc(m.get('id', '—'))}</td>"
            f"<td>{esc(m.get('label') or '')}</td>"
            + _num_cell(m.get("mass_before"))
            + _num_cell(m.get("mass_after"))
            + _num_cell(delta, delta_cls)
            + _num_cell(m.get("area_px_before"))
            + _num_cell(m.get("area_px_after"))
            + _num_cell(m.get("area_px_change"))
            + _num_cell(m.get("relative_density_before"))
            + _num_cell(m.get("relative_density_after"))
            + "</tr>"
        )
    return (
        '<table class="data"><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        '<p class="muted small">delta_pp = 100 × (mass B − mass A)，百分点；数值原样取自 '
        "comparison.json（仅显示舍入）。A/B 面积不同时同时看 Δarea_px，"
        "避免把面积扩大带来的占比增长解释成效率提升。delta_pp 只说明模型预测分布"
        "发生变化，不表示点击率或任务效率因而提高。</p>"
    )


def _single_side_table(rows: list, side_name: str) -> str:
    if not rows:
        return f'<p class="muted">无（{side_name}侧无此类区域）。</p>'
    head = (
        "<th>id</th><th>label</th><th>role</th><th>source</th><th>status</th>"
        "<th>area_px</th><th>probability_mass</th><th>relative_density</th>"
    )
    body = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        status = r.get("status")
        badge = f'<span class="badge {esc(status)}">{esc(status)}</span>' if status else "—"
        body.append(
            "<tr>"
            f"<td>{esc(r.get('id', '—'))}</td>"
            f"<td>{esc(r.get('label') or '')}</td>"
            f"<td>{esc(r.get('role') or '')}</td>"
            f"<td>{esc(r.get('source') or '—')}</td>"
            f"<td>{badge}</td>"
            + _num_cell(r.get("area_px"))
            + _num_cell(r.get("probability_mass"))
            + _num_cell(r.get("relative_density"))
            + "</tr>"
        )
    return '<table class="data"><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def _side_stage(
    side: dict, suffix: str, colorscale_params: dict
) -> str:
    """一侧的查看器（原图/共用色阶叠加图切换 + 区域静态 SVG）。"""
    width, height = side.get("image_size") or (0, 0)
    regions = side.get("regions") or []
    base_uri = _data_uri(side["base_png_bytes"])
    overlay_uri = _data_uri(side["overlay_png_bytes"])
    analysis_id = side.get("analysis_id", "—")
    return (
        '<div class="stage-col">'
        f'<h3>{esc(side.get("title") or ("A（before）" if suffix == "a" else "B（after）"))}'
        f' <span class="muted small mono">{esc(analysis_id)}</span></h3>'
        '<div class="viewbar">'
        f'<label class="seg" for="vm-{suffix}-overlay">'
        f'<input type="radio" name="view-mode-{suffix}" id="vm-{suffix}-overlay" value="overlay" checked>'
        "叠加</label>"
        f'<label class="seg" for="vm-{suffix}-original">'
        f'<input type="radio" name="view-mode-{suffix}" id="vm-{suffix}-original" value="original">'
        "原图</label>"
        f'<label class="small muted"><input type="checkbox" id="toggle-regions-{suffix}" checked> 区域框</label>'
        "</div>"
        f'<div class="stage" id="stage-{suffix}" data-mode="overlay">'
        f'<img class="layer base" alt="原图 {esc(suffix.upper())}" src="{base_uri}">'
        f'<img class="layer abs overlay" alt="共用色阶叠加 {esc(suffix.upper())}" src="{overlay_uri}">'
        f'<div class="svg-host">{_regions_svg(regions, width, height)}</div>'
        "</div>"
        f'<p class="muted small">{int(width)}×{int(height)}；叠加图使用 A/B 共用色阶 '
        f'<code>{esc(json.dumps(colorscale_params, ensure_ascii=False))}</code></p>'
        "</div>"
    )


def render_comparison_html(
    *,
    comparison: dict,
    side_a: dict,
    side_b: dict,
    colorscale_params: dict,
    overlay_alpha: float,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """低层入口：组装 A/B 对比页 HTML 并写盘；返回输出路径。

    ``side_a`` / ``side_b`` 结构::

        {"analysis_id": str, "base_png_bytes": bytes, "overlay_png_bytes": bytes,
         "regions": list[dict], "image_size": (width, height), "title"?: str}

    overlay_png_bytes 必须已由 :func:`render_shared_overlays`（或等价方式）
    以 ``colorscale_params`` 同一份参数渲染——共用色阶由调用链保证，
    本页把该参数展示并（经 C1）写入 manifest。
    """
    css_text = _load_resource("report.css")
    js_text = _load_resource("compare.js")
    if title is None:
        before_id = (comparison.get("before") or {}).get("analysis_id", "?")
        after_id = (comparison.get("after") or {}).get("analysis_id", "?")
        title = f"UI Attention A/B 对比 {before_id} → {after_id}"

    compat = _compat(comparison)
    compatible = compat.get("compatible", False)
    mismatches = compat.get("mismatches") or []
    canvas_note = compat.get("canvas_note") or ""
    exploratory = bool(compat.get("exploratory", False))
    exploratory_reasons = compat.get("exploratory_reasons") or []
    matched = _region_rows(comparison, "matched")
    added = _region_rows(comparison, "added")
    removed = _region_rows(comparison, "removed")
    limitations = comparison.get("limitations") or []

    shared_scale = ColorScale.from_params(colorscale_params)
    bar_uri = _data_uri(colorbar_png_bytes(shared_scale))

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="zh-CN">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src data:; "
        f"style-src {_csp_hash(css_text)}; "
        f"script-src {_csp_hash(js_text)}; "
        'form-action \'none\'; base-uri \'none\'">'
    )
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{esc(title)}</title>")
    parts.append(f"<style>{css_text}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")

    parts.append(f"<h1>{esc(title)}</h1>")
    before_id = (comparison.get("before") or {}).get("analysis_id", "—")
    after_id = (comparison.get("after") or {}).get("analysis_id", "—")
    parts.append(
        '<div class="viewbar">'
        '<span class="badge evidence">model_prediction</span>'
        f'<span class="muted small">schema:</span> <code>{esc(comparison.get("schema_version", "—"))}</code>'
        f'<span class="muted small">A(before):</span> <code>{esc(before_id)}</code>'
        f'<span class="muted small">B(after):</span> <code>{esc(after_id)}</code>'
        "</div>"
    )
    parts.append(
        '<div class="notice warn small">delta_pp 只说明<b>模型预测分布</b>发生变化，'
        "不表示点击率或任务效率因而提高（validation-plan §5）。</div>"
    )

    # ---- 兼容性
    parts.append("<h2>配置兼容性</h2>")
    if compatible:
        parts.append(
            '<div class="notice">兼容性检查通过：模型、权重、预处理、中心偏置、'
            "观看配置、指标版本与画布尺寸一致。</div>"
        )
    else:
        parts.append(
            '<div class="notice error"><b>不兼容：拒绝正式比较</b>（CLI 侧对应退出码 6 '
            "COMPARISON_INCOMPATIBLE）。以下为保留的原因；本页不展示差值表。</div>"
        )
        rows = "".join(
            "<tr>"
            f"<td>{esc(m.get('field', '—'))}</td>"
            f"<td>{esc(m.get('description', ''))}</td>"
            f"<td>{_render_value(m.get('before'))}</td>"
            f"<td>{_render_value(m.get('after'))}</td>"
            "</tr>"
            for m in mismatches
            if isinstance(m, dict)
        )
        parts.append(
            '<table class="data"><thead><tr><th>field</th><th>说明</th><th>before</th><th>after</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    if canvas_note:
        parts.append(f'<p class="muted small">{esc(canvas_note)}</p>')
    if exploratory:
        parts.append(
            '<div class="notice warn"><b>探索性比较</b>：画面状态、任务或内容明显不同，'
            "不作修改的因果结论。原因：" + esc("；".join(str(r) for r in exploratory_reasons)) + "</div>"
        )

    # ---- 共用色阶
    parts.append("<h2>A/B 共用色阶</h2>")
    parts.append(
        '<div class="panel"><div class="colorbar-wrap">'
        f'<img class="colorbar" alt="共用色标条" src="{bar_uri}">'
        '<div class="colorbar-axis">'
        f"<span>vmin = {esc(format_metric(shared_scale.vmin))}</span>"
        f"<span>{esc(shared_scale.name)} / {esc(shared_scale.normalization)}</span>"
        f"<span>vmax = {esc(format_metric(shared_scale.vmax))}</span>"
        "</div></div>"
        "<p class=\"muted small\">两侧叠加图由<b>同一份</b>色阶参数渲染（不各自拉满）；"
        "参数随 manifest 记录：<code>"
        + esc(json.dumps({"colorscale": colorscale_params, "overlay_alpha": overlay_alpha}, ensure_ascii=False))
        + "</code>。色标条仅用于读图，统计数值一律以 comparison.json / analysis.json 浮点值为准，"
        "不得从 PNG 颜色反推。</p></div>"
    )

    # ---- 并排视图
    parts.append("<h2>并排视图</h2>")
    parts.append('<div class="stage-wrap">')
    parts.append(_side_stage(side_a, "a", colorscale_params))
    parts.append(_side_stage(side_b, "b", colorscale_params))
    parts.append("</div>")

    # ---- delta 表 / 新增 / 移除
    if compatible:
        parts.append("<h2>配对区域差值（delta_pp）</h2>")
        parts.append(_matched_table(matched))
        parts.append("<h2>新增区域（仅 B 侧存在）</h2>")
        parts.append(_single_side_table(added, "新增"))
        parts.append("<h2>移除区域（仅 A 侧存在）</h2>")
        parts.append(_single_side_table(removed, "移除"))
        parts.append(
            '<p class="muted small">新增/移除区域单独记录：不编造缺失一侧的 0 值，'
            "不参与差值计算；不匹配区域不计算差值（data-contract §6）。</p>"
        )
    else:
        parts.append("<h2>配对区域差值</h2>")
        parts.append('<p class="muted">不兼容比较不展示数值差（原因见上）。</p>')

    if limitations:
        parts.append("<h2>limitations</h2>")
        parts.append(_render_value(list(limitations)))

    parts.append("<footer>")
    parts.append(
        "<p>渲染参数（写入 manifest）：<pre>"
        + esc(json.dumps(overlay_record(shared_scale, overlay_alpha), ensure_ascii=False, indent=2))
        + "</pre></p>"
    )
    parts.append(
        "<p>自包含页面：无外部脚本/样式/字体，无自动网络请求（CSP default-src 'none'）。"
        "嵌入截图的 HTML 属截图数据，不纳入公开仓库。</p>"
    )
    parts.append("</footer>")

    parts.append("</main>")
    # 对比页当前无浏览器内编辑；数据岛保留给未来联动（内容与展示一致、转义安全）
    parts.append(
        '<script type="application/json" id="report-data">'
        + json_data_island(
            {
                "colorscale": colorscale_params,
                "overlay_alpha": overlay_alpha,
                "before_analysis_id": (comparison.get("before") or {}).get("analysis_id"),
                "after_analysis_id": (comparison.get("after") or {}).get("analysis_id"),
            }
        )
        + "</script>"
    )
    parts.append(f"<script>{js_text}</script>")
    parts.append("</body>")
    parts.append("</html>")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8", newline="\n")
    return out


def render_compare_report(
    *,
    out_dir: Path | str,
    comparison: dict,
    base_a: np.ndarray,
    density_a: np.ndarray,
    regions_a: list[dict],
    base_b: np.ndarray,
    density_b: np.ndarray,
    regions_b: list[dict],
    name: str = "ember",
    normalization: str = "linear",
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    title: str | None = None,
) -> dict[str, Path]:
    """编排入口：共用色阶渲染两侧 + 组装对比页；返回产物路径字典。

    返回键：``overlay_a / heatmap_a / overlay_b / heatmap_b / report``。
    ``regions_a`` / ``regions_b`` 为两侧标注行（RegionSpec.to_dict 形态），
    用于区域框静态渲染；density 必须已是对应原图尺寸。
    """
    out = Path(out_dir)
    render = render_shared_overlays(
        base_a=base_a, density_a=density_a, base_b=base_b, density_b=density_b,
        out_dir=out, name=name, normalization=normalization, alpha=alpha,
    )

    def _png_bytes(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(arr), mode="RGB").save(buf, format="PNG")
        return buf.getvalue()

    base_a_arr = np.asarray(base_a)
    base_b_arr = np.asarray(base_b)
    side_a = {
        "analysis_id": (comparison.get("before") or {}).get("analysis_id", "—"),
        "title": "A（before）",
        "base_png_bytes": _png_bytes(base_a_arr),
        "overlay_png_bytes": render.overlay_a.read_bytes(),
        "regions": regions_a,
        "image_size": (base_a_arr.shape[1], base_a_arr.shape[0]),
    }
    side_b = {
        "analysis_id": (comparison.get("after") or {}).get("analysis_id", "—"),
        "title": "B（after）",
        "base_png_bytes": _png_bytes(base_b_arr),
        "overlay_png_bytes": render.overlay_b.read_bytes(),
        "regions": regions_b,
        "image_size": (base_b_arr.shape[1], base_b_arr.shape[0]),
    }
    report_path = render_comparison_html(
        comparison=comparison,
        side_a=side_a,
        side_b=side_b,
        colorscale_params=render.colorscale_params,
        overlay_alpha=alpha,
        out_path=out / "report.html",
        title=title,
    )
    return {
        "overlay_a": render.overlay_a,
        "heatmap_a": render.heatmap_a,
        "overlay_b": render.overlay_b,
        "heatmap_b": render.heatmap_b,
        "report": report_path,
    }
