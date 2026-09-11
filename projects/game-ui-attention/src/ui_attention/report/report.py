"""report.html 生成：自包含、全转义、CSP 收紧的本地报告页（单图）。

安全与自包含口径（data-contract §5、technical-design §6/§9、validation-plan
§1"文本呈现"行）：

- **自包含**：CSS/JS 资源（html/ 目录，本包自带）内联进单文件；原图/热图/
  叠加图以 ``data:image/png;base64`` 内嵌；无任何外部脚本、样式、字体或
  自动网络请求；产物中不出现 ``http://`` / ``https://`` / ``<script src``
  字面量（SVG 命名空间由 HTML 解析器处理，不写 xmlns URL）。
- **CSP 收紧**：``default-src 'none'; img-src data:; style-src 'sha256-…';
  script-src 'sha256-…'; form-action 'none'; base-uri 'none'``——内联
  style/script 以内容哈希放行，无 ``unsafe-inline``；无 connect-src，
  fetch/XHR 被 CSP 拒绝。
- **转义**：区域名/玩家目标/评审文本等全部用户可控文本经 HTML 转义后进入
  标记；交互数据经 ``<script type="application/json">`` 数据岛传递
  （``ensure_ascii`` + ``</`` 转义，JSON 数据岛不被解析为 HTML）。
- 嵌入原图的 HTML 属截图数据，不纳入公开仓库（由运行目录/gitignore 策略
  保证，页面 footer 亦有提示）。
- 指标数值只从传入的 analysis dict **原样展示**（仅做显示舍入），本模块
  不重新计算任何统计，不从 PNG 颜色反推数值。
- candidate 状态区域以虚线边框 + "候选边界"徽章标识（technical-design §6）。
- 区域圈选编辑与导出在浏览器内完成（Blob 下载），不隐式写回本地文件。

data-contract §4 的 analysis.json 部分子字段名尚未被 C1 contracts/ 冻结
（如 mass 键名），本模块以文档 schema 编码并用防御性取键（候选键列表），
缺失字段渲染为"—"；C1 契约冻结后如有出入由总控协调对齐（单点修改
``_pick`` 调用处）。

渲染入口（纯函数式：输入数据 + 配置 → 输出文件路径）：

- ``render_report_html(*, analysis, regions_payload, base_png_bytes,
  heatmap_png_bytes, overlay_png_bytes, image_size, image_sha256,
  colorscale_params, overlay_alpha, review=None, out_path,
  title=None, player_goal=None) -> Path``
  低层入口：只写 out_path 一个 HTML 文件，图片以 bytes 传入。
- ``render_run_report(*, out_dir, base_image, density, analysis,
  regions_payload, colorscale, alpha=DEFAULT_OVERLAY_ALPHA, review=None,
  base_png_path=None, title=None) -> dict[str, Path]``
  编排入口：由原图 ndarray + 密度 ndarray 渲染 overlay.png / heatmap.png
  并组装 report.html，返回 ``{"overlay", "heatmap", "report"}`` 路径字典
  （density 必须已是原图尺寸；重采样属 C1 imaging 职责）。
"""

from __future__ import annotations

import base64
import hashlib
import html as _html
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .colorscale import ColorScale
from .export_regions import REGIONS_SCHEMA_VERSION
from .overlay import (
    DEFAULT_OVERLAY_ALPHA,
    overlay_record,
    render_heatmap_png,
    render_overlay_png,
)

__all__ = [
    "build_report_html",
    "render_report_html",
    "render_run_report",
]

_HTML_DIR = Path(__file__).resolve().parent / "html"


def _load_resource(name: str) -> str:
    return (_HTML_DIR / name).read_text(encoding="utf-8")


def esc(value: Any) -> str:
    """HTML 转义（含引号），所有用户可控文本入标记前必经此函数。"""
    return _html.escape(str(value), quote=True)


def _csp_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def _data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def json_data_island(obj: Any) -> str:
    """序列化为可安全嵌入 ``<script type="application/json">`` 的文本。

    ensure_ascii=True 消除 U+2028/2029 与非 ASCII 字节序问题；``<`` / ``>``
    整体替换为 JSON 转义 ``\\u003c`` / ``\\u003e``（解析后还原），保证岛内
    不含任何原始尖括号——既不可能提前闭合 script 数据块，也不可能被
    解析为标记（深度防御，测试断言岛内无 ``<`` 字符）。
    """
    text = json.dumps(obj, ensure_ascii=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def _pick(d: Any, *names: str) -> Any:
    """按候选键名顺序取第一个存在的值（防御 C1 契约字段名微调）。"""
    if not isinstance(d, dict):
        return None
    for n in names:
        if n in d:
            return d[n]
    return None


def format_metric(value: Any) -> str:
    """指标显示格式化（仅显示舍入，不改变数值；完整值放 title 属性）。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "非有限值"
        return f"{value:.6g}"
    return esc(value)


def _metric_title(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _render_value(value: Any) -> str:
    """通用防御渲染：dict → dl，list → ul，标量 → code（全转义）。"""
    if isinstance(value, dict):
        if not value:
            return '<span class="muted">（空）</span>'
        parts = ['<dl class="meta">']
        for k, v in value.items():
            parts.append(f"<dt>{esc(k)}</dt><dd>{_render_value(v)}</dd>")
        parts.append("</dl>")
        return "".join(parts)
    if isinstance(value, (list, tuple)):
        if not value:
            return '<span class="muted">（空）</span>'
        items = "".join(f"<li>{_render_value(v)}</li>" for v in value)
        return f'<ul class="plain">{items}</ul>'
    if isinstance(value, float):
        return f"<code>{esc(format_metric(value))}</code>"
    if value is None:
        return '<span class="muted">—</span>'
    return f"<code>{esc(json.dumps(value, ensure_ascii=False))}</code>"


def _status_badge(value: Any) -> str:
    v = str(value)
    cls = {
        "complete": "status-complete",
        "confirmed": "confirmed",
        "candidate": "candidate",
        "failed": "status-failed",
        "pending": "status-pending",
    }.get(v, "")
    return f'<span class="badge {cls}">{esc(v)}</span>'


def _regions_svg(regions: list[dict], width: int, height: int) -> str:
    """服务端静态渲染区域层 SVG（无 JS 也可见；无 xmlns URL 字面量）。

    candidate 区域用虚线边框类（technical-design §6"展示边框"）。
    """
    parts = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none"><g>'
    ]
    for region in regions:
        if not isinstance(region, dict):
            continue
        geom = region.get("geometry")
        if not isinstance(geom, dict):
            continue
        status = region.get("status")
        cls = "region-box candidate" if status == "candidate" else "region-box confirmed"
        suffix = "（候选）" if status == "candidate" else ""
        label = region.get("label") or region.get("id") or ""
        gtype = geom.get("type")
        if gtype == "rect" and all(
            isinstance(geom.get(k), (int, float)) for k in ("x", "y", "width", "height")
        ):
            x, y = geom["x"], geom["y"]
            rw, rh = geom["width"], geom["height"]
            parts.append(
                f'<rect class="{cls}" x="{esc(x)}" y="{esc(y)}" '
                f'width="{esc(rw)}" height="{esc(rh)}"></rect>'
            )
            parts.append(
                f'<text class="region-label" x="{esc(x + 4)}" y="{esc(y + 16)}">{esc(label)}{suffix}</text>'
            )
        elif gtype == "polygon" and isinstance(geom.get("points"), list) and geom["points"]:
            pts = geom["points"]
            try:
                pt_str = " ".join(f"{p[0]},{p[1]}" for p in pts)
            except (TypeError, IndexError, KeyError):
                continue
            parts.append(f'<polygon class="{cls}" points="{esc(pt_str)}"></polygon>')
            try:
                px, py = pts[0][0], pts[0][1]
                parts.append(
                    f'<text class="region-label" x="{esc(px + 4)}" y="{esc(py + 16)}">{esc(label)}{suffix}</text>'
                )
            except (TypeError, IndexError, KeyError):
                pass
    parts.append("</g></svg>")
    return "".join(parts)


def _metrics_table(analysis_regions: Any) -> str:
    """区域指标表：数值全部原样来自 analysis.json 的 regions 字段。

    列键对齐 C1 冻结 schema（contracts/analysis.py _REGION_RESULT_KEYS）：
    id / label / role / source / status / area_px / area_fraction /
    probability_mass / relative_density。
    """
    if not isinstance(analysis_regions, list) or not analysis_regions:
        return '<p class="muted">analysis.json 未提供区域指标（regions 为空或缺失）。</p>'
    cols = [
        ("id", ("id",)),
        ("label", ("label",)),
        ("role", ("role",)),
        ("source", ("source",)),
        ("status", ("status",)),
        ("area_px", ("area_px",)),
        ("area_fraction", ("area_fraction",)),
        ("probability_mass", ("probability_mass",)),
        ("relative_density", ("relative_density",)),
    ]
    head = "".join(f"<th>{esc(name)}</th>" for name, _ in cols)
    body_rows = []
    for row in analysis_regions:
        if not isinstance(row, dict):
            continue
        cells = []
        for _, keys in cols:
            value = _pick(row, *keys)
            if keys[0] == "status":
                cells.append(f"<td>{_status_badge(value) if value is not None else '—'}</td>")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(
                    f'<td class="num" title="{esc(_metric_title(value))}">{esc(format_metric(value))}</td>'
                )
            else:
                cells.append(f"<td>{esc(value) if value is not None else '—'}</td>")
        status = _pick(row, "status")
        tr_cls = ' class="row-candidate"' if status == "candidate" else ""
        body_rows.append(f"<tr{tr_cls}>{''.join(cells)}</tr>")
    return (
        '<table class="data"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
        '<p class="muted small">数值原样取自 analysis.json（仅显示舍入，悬停查看完整值）；'
        "本页不重新计算统计。probability_mass 为预测注视分布落入区域的质量占比，"
        "不表示真实玩家观看比例。</p>"
    )


def _review_section(review: dict | None) -> str:
    if review is None:
        return '<p class="muted">未提供评审（review.json 缺省）。</p>'
    findings = review.get("findings") if isinstance(review, dict) else None
    if not isinstance(findings, list) or not findings:
        return '<p class="muted">review.json 无 findings 记录。</p>'
    blocks = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        fid = _pick(f, "id") or "(无 id)"
        etype = _pick(f, "evidence_type") or "(未标注)"
        rids = _pick(f, "region_ids") or []
        rids_txt = ", ".join(str(r) for r in rids) if isinstance(rids, list) else str(rids)
        refs = _pick(f, "evidence_refs") or []
        refs_txt = ", ".join(str(r) for r in refs) if isinstance(refs, list) else str(refs)
        sec = []
        for key, zh in (
            ("observation", "观察"),
            ("inference", "推断"),
            ("recommendation", "建议"),
            ("validation_needed", "需验证"),
        ):
            val = _pick(f, key)
            if val is not None:
                sec.append(f'<div class="fsec"><b>{zh}</b>{esc(val)}</div>')
        blocks.append(
            '<div class="finding">'
            f'<div class="fhead"><span class="badge">{esc(fid)}</span>'
            f'<span class="badge evidence">{esc(etype)}</span>'
            f'<span class="muted small">区域: {esc(rids_txt) or "—"}</span></div>'
            + "".join(sec)
            + (f'<div class="fsec small muted">证据引用: {esc(refs_txt)}</div>' if refs_txt else "")
            + "</div>"
        )
    return "".join(blocks)


def build_report_html(
    *,
    analysis: dict,
    regions_payload: dict,
    base_png_bytes: bytes,
    heatmap_png_bytes: bytes | None,
    overlay_png_bytes: bytes | None,
    image_size: tuple[int, int],
    image_sha256: str,
    colorscale_params: dict,
    overlay_alpha: float,
    review: dict | None = None,
    title: str | None = None,
    player_goal: str | None = None,
) -> str:
    """组装完整 report.html 文本（不落盘）。全部输入为数据 + 配置。"""
    width, height = int(image_size[0]), int(image_size[1])
    regions = regions_payload.get("regions") if isinstance(regions_payload, dict) else None
    if not isinstance(regions, list):
        regions = []
    css_text = _load_resource("report.css")
    js_text = _load_resource("report.js")

    if title is None:
        title = f"UI Attention 报告 {analysis.get('analysis_id', '')}".strip()
    if player_goal is None:
        # C1 冻结 schema：player_goal 为 analysis 顶层扩展字段（contracts/analysis.py）
        player_goal = analysis.get("player_goal") if isinstance(analysis, dict) else None

    data_island = json_data_island(
        {
            "image": {
                "sha256": str(regions_payload.get("image_sha256") or image_sha256),
                "width": width,
                "height": height,
                "regions_schema_version": regions_payload.get(
                    "schema_version", REGIONS_SCHEMA_VERSION
                ),
            },
            "regions": regions,
        }
    )

    analysis_regions = analysis.get("regions") if isinstance(analysis, dict) else None
    input_meta = _pick(analysis, "input") or {}
    model_meta = _pick(analysis, "model") or {}
    profile_meta = _pick(analysis, "profile") or {}
    runtime_meta = _pick(analysis, "runtime") or {}
    limitations = _pick(analysis, "limitations")
    errors = _pick(analysis, "errors")
    artifacts = _pick(analysis, "artifacts")

    goal_html = (
        f'<div class="notice">玩家目标：<b>{esc(player_goal)}</b></div>'
        if player_goal
        else '<div class="notice warn">未提供玩家目标：评审只能是描述性的，'
        "不得声称目标完成路径合理（data-contract §2）。</div>"
    )

    rec = overlay_record(
        ColorScale.from_params(colorscale_params), overlay_alpha
    )

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

    # ---- header
    comp_status = analysis.get("computation_status", "(缺失)")
    rev_status = analysis.get("review_status", "(缺失)")
    ev_type = analysis.get("evidence_type", "model_prediction")
    parts.append(f"<h1>{esc(title)}</h1>")
    parts.append(
        '<div class="viewbar">'
        f'<span class="badge evidence">{esc(ev_type)}</span>'
        f'<span class="muted small">analysis_id:</span> <code>{esc(analysis.get("analysis_id", "—"))}</code>'
        f'<span class="muted small">schema:</span> <code>{esc(analysis.get("schema_version", "—"))}</code>'
        f'<span class="muted small">计算:</span> {_status_badge(comp_status)}'
        f'<span class="muted small">评审:</span> {_status_badge(rev_status)}'
        "</div>"
    )
    parts.append(
        '<div class="notice warn small">本页面全部热图与指标为<b>模型预测</b>渲染，'
        "不是真实眼动、点击率或可用性结论。</div>"
    )
    parts.append(goal_html)

    # ---- viewer
    parts.append("<h2>视图</h2>")
    parts.append(
        '<div class="viewbar">'
        '<label class="seg" for="vm-original">'
        '<input type="radio" name="view-mode" id="vm-original" value="original" checked>'
        "原图</label>"
        '<label class="seg" for="vm-heatmap">'
        '<input type="radio" name="view-mode" id="vm-heatmap" value="heatmap">'
        "热图</label>"
        '<label class="seg" for="vm-overlay">'
        '<input type="radio" name="view-mode" id="vm-overlay" value="overlay">'
        "叠加</label>"
        '<label class="small muted"><input type="checkbox" id="toggle-regions" checked> 显示区域框</label>'
        '<span class="muted small">热图 = 色阶全强度叠加；叠加 = 固定透明度 α 合成（记录于页脚）</span>'
        "</div>"
    )
    parts.append('<div class="stage-wrap">')
    parts.append('<div class="stage-col">')
    parts.append('<div class="stage" id="main-stage" data-mode="original">')
    parts.append(f'<img class="layer base" id="base-layer" alt="原图" src="{_data_uri(base_png_bytes)}">')
    if heatmap_png_bytes is not None:
        parts.append(f'<img class="layer abs heat" alt="热图" src="{_data_uri(heatmap_png_bytes)}">')
    if overlay_png_bytes is not None:
        parts.append(f'<img class="layer abs overlay" alt="叠加图" src="{_data_uri(overlay_png_bytes)}">')
    parts.append(f'<div class="svg-host" id="region-svg-host">{_regions_svg(regions, width, height)}</div>')
    parts.append("</div>")  # stage
    parts.append(f'<p class="muted small">原图 {width}×{height}，SHA-256 <code>{esc(image_sha256)}</code></p>')
    parts.append("</div>")  # stage-col

    # ---- edit panel
    parts.append('<div class="side-col">')
    parts.append('<div class="panel">')
    parts.append("<h3>区域圈选与编辑</h3>")
    parts.append(
        '<div class="btnrow">'
        '<button class="btn active" id="mode-off" type="button">浏览</button>'
        '<button class="btn" id="mode-rect" type="button">矩形拖选</button>'
        '<button class="btn" id="mode-poly" type="button">多边形点选</button>'
        '<button class="btn" id="btn-finish-poly" type="button" disabled>完成多边形</button>'
        '<button class="btn" id="btn-clear-draft" type="button">清除草稿</button>'
        "</div>"
    )
    parts.append(
        '<div class="fieldrow"><label for="f-id">id</label>'
        '<input type="text" id="f-id" autocomplete="off"></div>'
        '<div class="fieldrow"><label for="f-label">label</label>'
        '<input type="text" id="f-label" autocomplete="off"></div>'
        '<div class="fieldrow"><label for="f-role">role</label>'
        '<input type="text" id="f-role" autocomplete="off"></div>'
        '<div class="fieldrow"><label for="f-source">source</label>'
        '<select id="f-source"><option value="manual" selected>manual</option>'
        '<option value="agent">agent</option><option value="imported">imported</option></select></div>'
        '<div class="fieldrow"><label for="f-status">status</label>'
        '<select id="f-status"><option value="confirmed" selected>confirmed</option>'
        '<option value="candidate">candidate</option></select></div>'
    )
    parts.append(
        '<div class="btnrow">'
        '<button class="btn primary" id="btn-add" type="button">添加区域</button>'
        '<button class="btn" id="btn-cancel-edit" type="button" disabled>取消编辑</button>'
        "</div>"
    )
    parts.append('<div id="edit-note" class="small muted"></div>')
    parts.append('<div id="dirty-note" class="dirty-note"></div>')
    parts.append(
        '<div class="btnrow"><button class="btn primary" id="btn-export" type="button">'
        "导出 regions JSON（浏览器下载）</button></div>"
    )
    parts.append('<div id="export-note" class="small muted"></div>')
    parts.append(
        '<p class="muted small">导出文件携带图片 SHA-256 绑定；本页面不写回本地文件、'
        "不联系外部服务。把下载的 JSON 交给 <code>ui-attention summarize</code> "
        "重算统计（不重新推理）。</p>"
    )
    parts.append("</div>")  # panel
    parts.append("</div>")  # side-col
    parts.append("</div>")  # stage-wrap

    # ---- regions annotation table (editable, JS 重建)
    parts.append("<h2>区域标注</h2>")
    parts.append(
        '<table class="data"><thead><tr><th>id</th><th>label</th><th>role</th>'
        "<th>source</th><th>status</th><th>geometry</th><th>操作</th></tr></thead>"
        '<tbody id="regions-tbody"></tbody></table>'
    )
    parts.append(
        '<p class="muted small">黄色虚线边框 = <span class="badge candidate">candidate</span> '
        "候选边界（Agent 自动识别，未经人工确认）；相关指标与结论同样基于候选边界。</p>"
    )

    # ---- metrics
    parts.append("<h2>区域指标（来自 analysis.json）</h2>")
    parts.append(_metrics_table(analysis_regions))

    # ---- metadata
    parts.append("<h2>输入与运行证据</h2>")
    parts.append("<h3>input</h3>")
    parts.append(_render_value(input_meta))
    parts.append("<h3>model</h3>")
    parts.append(_render_value(model_meta))
    parts.append("<h3>profile</h3>")
    parts.append(_render_value(profile_meta))
    parts.append("<h3>runtime</h3>")
    parts.append(_render_value(runtime_meta))
    parts.append("<h3>limitations</h3>")
    parts.append(_render_value(limitations))
    parts.append("<h3>errors</h3>")
    parts.append(_render_value(errors))
    parts.append("<h3>artifacts</h3>")
    parts.append(_render_value(artifacts))

    # ---- review
    parts.append("<h2>语义评审（review.json）</h2>")
    parts.append(_review_section(review))

    # ---- footer
    parts.append("<footer>")
    parts.append(
        "<p>渲染参数（可写入 manifest）：<pre>"
        + esc(json.dumps({"overlay": rec}, ensure_ascii=False, indent=2))
        + "</pre></p>"
    )
    parts.append(
        "<p>自包含页面：无外部脚本/样式/字体，无自动网络请求（CSP default-src 'none'）。"
        "嵌入原图的 HTML 属截图数据，不纳入公开仓库。</p>"
    )
    parts.append("</footer>")

    parts.append("</main>")
    parts.append(f'<script type="application/json" id="report-data">{data_island}</script>')
    parts.append(f"<script>{js_text}</script>")
    parts.append("</body>")
    parts.append("</html>")
    return "".join(parts)


def render_report_html(
    *,
    analysis: dict,
    regions_payload: dict,
    base_png_bytes: bytes,
    heatmap_png_bytes: bytes | None,
    overlay_png_bytes: bytes | None,
    image_size: tuple[int, int],
    image_sha256: str,
    colorscale_params: dict,
    overlay_alpha: float,
    review: dict | None = None,
    out_path: Path | str,
    title: str | None = None,
    player_goal: str | None = None,
) -> Path:
    """低层渲染入口：组装 report.html 并写盘；返回输出路径。"""
    text = build_report_html(
        analysis=analysis,
        regions_payload=regions_payload,
        base_png_bytes=base_png_bytes,
        heatmap_png_bytes=heatmap_png_bytes,
        overlay_png_bytes=overlay_png_bytes,
        image_size=image_size,
        image_sha256=image_sha256,
        colorscale_params=colorscale_params,
        overlay_alpha=overlay_alpha,
        review=review,
        title=title,
        player_goal=player_goal,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


def _png_bytes_from_array(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def render_run_report(
    *,
    out_dir: Path | str,
    base_image: np.ndarray,
    density: np.ndarray,
    analysis: dict,
    regions_payload: dict,
    colorscale: ColorScale,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    review: dict | None = None,
    base_png_path: Path | str | None = None,
    title: str | None = None,
) -> dict[str, Path]:
    """编排入口：一次调用产出 overlay.png、heatmap.png、report.html。

    - ``base_image``：方向已处理的原图 ``(H, W, 3)`` uint8。
    - ``density``：``(H, W)`` float64，**必须已是原图尺寸**（推理分辨率 →
      原图尺寸的重采样与再归一化属 C1 imaging/shape_mapping 职责）。
    - ``base_png_path``：若运行目录已有原图 PNG（C1 落盘），直接嵌入其
      字节；否则由 base_image 确定性编码 PNG 字节用于内嵌。
    - 返回 ``{"overlay": Path, "heatmap": Path, "report": Path}``。
    - image_sha256 优先取 regions_payload 的绑定值；缺失时要求 C1 在
      analysis.input 中提供（防御性取键），两者皆无则 ValueError——
      不允许无绑定渲染。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = np.asarray(base_image)
    if base.ndim != 3 or base.shape[2] != 3 or base.dtype != np.uint8:
        raise ValueError(f"base_image must be (H, W, 3) uint8, got {base.shape} {base.dtype}")
    height, width = base.shape[:2]

    image_sha256 = regions_payload.get("image_sha256") if isinstance(regions_payload, dict) else None
    if not image_sha256:
        image_sha256 = _pick(_pick(analysis, "input") or {}, "image_sha256", "sha256", "image_hash")
    if not image_sha256:
        raise ValueError(
            "image SHA-256 binding is required (regions_payload.image_sha256 "
            "or analysis.input.image_sha256)"
        )

    overlay_path = render_overlay_png(
        base_image=base,
        density=density,
        colorscale=colorscale,
        alpha=alpha,
        out_path=out / "overlay.png",
    )
    heatmap_path = render_heatmap_png(
        density=density, colorscale=colorscale, out_path=out / "heatmap.png"
    )

    if base_png_path is not None:
        base_png_bytes = Path(base_png_path).read_bytes()
    else:
        base_png_bytes = _png_bytes_from_array(base)

    report_path = render_report_html(
        analysis=analysis,
        regions_payload=regions_payload,
        base_png_bytes=base_png_bytes,
        heatmap_png_bytes=heatmap_path.read_bytes(),
        overlay_png_bytes=overlay_path.read_bytes(),
        image_size=(width, height),
        image_sha256=str(image_sha256),
        colorscale_params=colorscale.to_params(),
        overlay_alpha=alpha,
        review=review,
        out_path=out / "report.html",
        title=title,
    )
    return {"overlay": overlay_path, "heatmap": heatmap_path, "report": report_path}
