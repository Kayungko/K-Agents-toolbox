"""C3：HTML 报告单测（report.py / compare_view.py）。

覆盖任务验收项与 validation-plan §1：
- XSS 转义（区域名含 <script>、玩家目标含引号/HTML 实体、评审文本含实体）；
  区分"可见标记区"与"JSON 数据岛"分别断言。
- HTML 无外链（无 http(s)://、无 <script src、无 <link、无 fetch/XHR 等）。
- CSP 内容哈希收紧（default-src 'none' + sha256 重算一致）。
- 自包含（图片全部 data URI；CSS/JS 内联）。
- candidate 状态边框标识；指标数值原样来自 analysis（不重新计算）。
- A/B 对比页：共用色阶参数呈现、delta_pp 表、新增/移除单列区、
  不兼容时拒绝展示差值。
- 渲染确定性（同输入同输出文本）。

合成图与夹具全部代码生成；analysis 夹具用 C1 冻结契约
AnalysisRecord.from_dict 复核合法性。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from ui_attention.contracts.analysis import AnalysisRecord
from ui_attention.report.colorscale import ColorScale
from ui_attention.report.compare_view import render_compare_report, render_comparison_html
from ui_attention.report.export_regions import make_regions_payload
from ui_attention.report.report import build_report_html, render_run_report

XSS_LABEL = "<script>alert('xss-label')</script>"
XSS_GOAL = "\"><img src=x onerror=alert('goal')>"
XSS_REVIEW = "Tom & Jerry <b>bold</b> \"double\" 'single' </script>"
EVIL_ID = 'evil"id'


def make_base(h: int = 48, w: int = 64) -> np.ndarray:
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(30, 200, w, dtype=np.uint8)[None, :]
    base[:, :, 2] = np.linspace(200, 30, h, dtype=np.uint8)[:, None]
    base[h // 4 : h // 2, w // 4 : w // 2] = (240, 220, 40)
    return base


def make_density(h: int = 48, w: int = 64, peak: float = 0.02) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.exp(-(((xx - w * 0.6) ** 2 + (yy - h * 0.5) ** 2) / (2 * (w * 0.12) ** 2)))
    return d / d.sum() * peak


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def base() -> np.ndarray:
    return make_base()


@pytest.fixture()
def density() -> np.ndarray:
    return make_density()


@pytest.fixture()
def image_sha(base) -> str:
    return hashlib.sha256(png_bytes(base)).hexdigest()


@pytest.fixture()
def regions_payload(image_sha) -> dict:
    return make_regions_payload(
        [
            {
                "id": "claim-button",
                "label": XSS_LABEL,
                "role": "primary-action",
                "geometry": {"type": "rect", "x": 10, "y": 8, "width": 20, "height": 6},
                "source": "manual",
                "status": "confirmed",
            },
            {
                "id": EVIL_ID,
                "label": "立绘 & 装饰",
                "role": "decoration",
                "geometry": {"type": "polygon", "points": [[30, 20], [60, 22], [45, 44]]},
                "source": "agent",
                "status": "candidate",
            },
        ],
        image_sha,
    )


@pytest.fixture()
def analysis(image_sha) -> dict:
    record = {
        "schema_version": "game-ui-attention-analysis/v1",
        "analysis_id": "an-20260911-001",
        "computation_status": "complete",
        "review_status": "complete",
        "evidence_type": "model_prediction",
        "metrics_version": "aoi-metrics/v1",
        "player_goal": XSS_GOAL,
        "input": {
            "image_path": "synthetic.png",
            "image_sha256": image_sha,
            "width": 64,
            "height": 48,
            "inference_width": 320,
            "inference_height": 240,
            "screen_type": "reward-summary",
        },
        "model": {
            "backend_id": "foveacast-onnx-3s",
            "version": "0.2.0",
            "weights_sha256": ["ab" * 32],
            "code_version": "c2-v1",
            "capabilities": ["spatial_density"],
        },
        "profile": {
            "profile_name": "foveacast-onnx-3s-v1",
            "config_hash": "cd" * 32,
            "preprocessing": {"input_size": [240, 320]},
            "centerbias": None,
            "viewing_conditions": {"assumption": "experimental"},
        },
        "runtime": {
            "dependencies": {"numpy": "2.5.3", "pillow": "12.3.0"},
            "device": "cpu",
            "precision": "fp32",
            "elapsed_ms": 123.4,
            "peak_mem_mb": 567.8,
        },
        "regions": [
            {
                "id": "claim-button",
                "label": XSS_LABEL,
                "role": "primary-action",
                "source": "manual",
                "status": "confirmed",
                "geometry": {"type": "rect", "x": 10, "y": 8, "width": 20, "height": 6},
                "area_px": 120,
                "area_fraction": 0.0390625,
                "probability_mass": 0.4321987654321,
                "relative_density": 11.06,
            },
            {
                "id": EVIL_ID,
                "label": "立绘 & 装饰",
                "role": "decoration",
                "source": "agent",
                "status": "candidate",
                "geometry": {"type": "polygon", "points": [[30, 20], [60, 22], [45, 44]]},
                "area_px": 313,
                "area_fraction": 0.10188802083333333,
                "probability_mass": 0.1875,
                "relative_density": 1.8403,
            },
        ],
        "limitations": ["合成图只证明计算链正确", "训练分布不含游戏 UI"],
        "artifacts": [{"path": "density.npy", "sha256": "ef" * 32, "size_bytes": 1234}],
        "errors": [],
    }
    AnalysisRecord.from_dict(record)  # 夹具必须符合 C1 冻结 schema，否则测试失败
    return record


@pytest.fixture()
def review() -> dict:
    return {
        "findings": [
            {
                "id": "F1",
                "region_ids": ["claim-button", EVIL_ID],
                "evidence_refs": ["analysis.regions[0].probability_mass"],
                "evidence_type": "computed",
                "observation": XSS_REVIEW,
                "inference": "立绘竞争 < 主按钮注意力（推断）",
                "recommendation": "保持当前层级；如需验证请做 A/B",
                "validation_needed": "真实眼动数据缺失，预测有效性未验证",
            }
        ]
    }


def render(analysis, regions_payload, base, density, review=None, **kw) -> str:
    cs = ColorScale("ember", float(density.min()), float(density.max()) * 1.0001)
    import tempfile
    from pathlib import Path

    from ui_attention.report.overlay import render_heatmap_png, render_overlay_png

    tmp = Path(tempfile.mkdtemp())
    render_overlay_png(
        base_image=base, density=density, colorscale=cs, alpha=0.55, out_path=tmp / "o.png"
    )
    render_heatmap_png(density=density, colorscale=cs, out_path=tmp / "h.png")
    return build_report_html(
        analysis=analysis,
        regions_payload=regions_payload,
        base_png_bytes=png_bytes(base),
        heatmap_png_bytes=(tmp / "h.png").read_bytes(),
        overlay_png_bytes=(tmp / "o.png").read_bytes(),
        image_size=(base.shape[1], base.shape[0]),
        image_sha256=regions_payload["image_sha256"],
        colorscale_params=cs.to_params(),
        overlay_alpha=0.55,
        review=review,
        **kw,
    )


def split_island(html_text: str) -> tuple[str, str, str]:
    """把页面拆成 (数据岛之前的标记, 数据岛内容, 数据岛之后的标记+JS)。"""
    marker = '<script type="application/json" id="report-data">'
    head, rest = html_text.split(marker)
    island, tail = rest.split("</script>", 1)
    return head, island, tail


class TestXssEscaping:
    def test_region_label_script_escaped_in_markup(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        head, island, tail = split_island(html_text)
        markup = head + tail
        # 可见标记区：不存在原始 <script> 载荷，存在转义形式
        assert "<script>alert" not in markup
        assert "&lt;script&gt;alert(&#x27;xss-label&#x27;)" in markup
        # 整页 script 开/闭标签恰好各 2 个（数据岛 + 内联 JS），标签文本未产生第三个脚本
        assert html_text.count("<script") == 2
        assert html_text.count("</script>") == 2

    def test_data_island_cannot_close_early(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        _, island, _ = split_island(html_text)
        # 岛内不得出现提前闭合序列；深度防御：整岛无原始尖括号（\u003c 转义）
        assert "</script" not in island
        assert "<" not in island and ">" not in island
        assert "\\u003c" in island
        data = json.loads(island)  # 岛内容仍是合法 JSON，转义可还原
        assert data["image"]["sha256"] == regions_payload["image_sha256"]
        assert any(r["id"] == EVIL_ID for r in data["regions"])
        labels = [r.get("label") for r in data["regions"]]
        assert XSS_LABEL in labels  # JSON 解析后原文还原（数据不损坏）

    def test_player_goal_quotes_escaped(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        head, _, tail = split_island(html_text)
        markup = head + tail
        assert "<img src=x" not in markup
        assert "onerror=alert('goal')" not in markup  # 原始引号形式不存在
        assert "&quot;&gt;&lt;img src=x onerror=alert(&#x27;goal&#x27;)&gt;" in markup

    def test_review_text_entities_escaped(self, analysis, regions_payload, base, density, review):
        html_text = render(analysis, regions_payload, base, density, review=review)
        head, _, tail = split_island(html_text)
        markup = head + tail
        assert "<b>bold</b>" not in markup
        assert "Tom &amp; Jerry &lt;b&gt;bold&lt;/b&gt;" in markup
        assert "&quot;double&quot;" in markup
        assert "&#x27;single&#x27;" in markup

    def test_evil_region_id_escaped_in_attributes(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        head, _, tail = split_island(html_text)
        markup = head + tail
        assert 'evil"id' not in markup
        assert "evil&quot;id" in markup


class TestNoExternalReferences:
    @pytest.mark.parametrize("page", ["report", "compare"])
    def test_no_links_no_network_apis(self, page, analysis, regions_payload, base, density, tmp_path):
        if page == "report":
            html_text = render(analysis, regions_payload, base, density)
        else:
            html_text = self._compare_html(tmp_path, base, density)
        lowered = html_text.lower()
        assert "http://" not in lowered
        assert "https://" not in lowered
        assert "<script src" not in lowered
        assert "<link" not in lowered
        assert "@import" not in lowered
        # CSS 不引用任何外部/内联资源（url(...) 只查 style 块：JS 的
        # URL.createObjectURL( 属本地 Blob API，非资源引用）
        style_body = re.search(r"<style>(.*?)</style>", html_text, re.S).group(1)
        assert "url(" not in style_body.lower()
        assert not re.search(r'<[^>]+\shref\s*=\s*["\']?(?:https?:)?//', html_text, re.I)
        for api in ("fetch(", "xmlhttprequest", "navigator.sendbeacon", "importscripts", "websocket", "eventsource"):
            assert api not in lowered, f"network API {api} must not appear"

    def _compare_html(self, tmp_path, base, density):
        comparison = _comparison_fixture()
        da = density
        db = make_density(peak=0.03)
        out = render_compare_report(
            out_dir=tmp_path / "cmp",
            comparison=comparison,
            base_a=base,
            density_a=da,
            regions_a=_side_regions(),
            base_b=base,
            density_b=db,
            regions_b=_side_regions(),
        )
        return out["report"].read_text(encoding="utf-8")

    def test_all_images_are_data_uris(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        srcs = re.findall(r'<img[^>]*?src="([^"]{1,40})', html_text)
        assert len(srcs) >= 3  # base + heat + overlay
        assert all(s.startswith("data:image/png;base64,") for s in srcs)


class TestCspAndSelfContainment:
    def test_csp_default_none_and_hash_matches_inline_blocks(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        csp = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', html_text)
        assert csp, "CSP meta 必须存在"
        policy = csp.group(1)
        assert "default-src 'none'" in policy
        assert "img-src data:" in policy
        assert "unsafe-inline" not in policy
        assert "base-uri 'none'" in policy
        # 内联 style/script 以内容哈希放行：重算哈希必须与 CSP 一致
        style_body = re.search(r"<style>(.*?)</style>", html_text, re.S).group(1)
        script_body = re.search(r"<script>(.*?)</script>", html_text, re.S).group(1)
        for body in (style_body, script_body):
            digest = base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode()
            assert f"'sha256-{digest}'" in policy

    def test_css_and_js_inlined(self, analysis, regions_payload, base, density):
        from ui_attention.report.report import _load_resource

        html_text = render(analysis, regions_payload, base, density)
        assert _load_resource("report.css") in html_text
        assert _load_resource("report.js") in html_text

    def test_render_deterministic(self, analysis, regions_payload, base, density):
        a = render(analysis, regions_payload, base, density)
        b = render(analysis, regions_payload, base, density)
        assert a == b


class TestPresentation:
    def test_candidate_border_and_badge(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        assert 'class="region-box candidate"' in html_text  # 虚线边框类
        assert 'class="region-box confirmed"' in html_text
        css = re.search(r"<style>(.*?)</style>", html_text, re.S).group(1)
        assert ".region-box.candidate" in css and "stroke-dasharray" in css
        assert "候选边界" in html_text

    def test_metrics_verbatim_from_analysis(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        # 显示舍入值 + title 中的完整原始值（数值来自 analysis，不重新计算）
        assert "0.432199" in html_text
        assert "0.4321987654321" in html_text  # title 完整值
        assert "11.06" in html_text
        assert "0.0390625" in html_text
        assert ">120<" in html_text  # area_px
        assert "不重新计算统计" in html_text

    def test_view_toggle_three_modes(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        for mode in ("original", "heatmap", "overlay"):
            assert f'value="{mode}"' in html_text
        assert 'id="main-stage" data-mode="original"' in html_text

    def test_export_flow_elements_present(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        assert 'id="btn-export"' in html_text
        assert "不会写回本地文件" in html_text
        assert "浏览器下载" in html_text
        assert 'id="mode-rect"' in html_text and 'id="mode-poly"' in html_text
        assert 'id="regions-tbody"' in html_text

    def test_evidence_disclaimers(self, analysis, regions_payload, base, density):
        html_text = render(analysis, regions_payload, base, density)
        assert "模型预测" in html_text
        assert "不是真实眼动" in html_text

    def test_missing_player_goal_warning(self, analysis, regions_payload, base, density):
        analysis_no_goal = dict(analysis)
        analysis_no_goal.pop("player_goal")
        html_text = render(analysis_no_goal, regions_payload, base, density)
        assert "未提供玩家目标" in html_text


class TestRenderRunReport:
    def test_artifacts_written_and_bound(self, analysis, regions_payload, base, density, tmp_path):
        cs = ColorScale("ember", float(density.min()), float(density.max()) * 1.0001)
        out = render_run_report(
            out_dir=tmp_path / "run",
            base_image=base,
            density=density,
            analysis=analysis,
            regions_payload=regions_payload,
            colorscale=cs,
            alpha=0.55,
        )
        assert set(out) == {"overlay", "heatmap", "report"}
        for p in out.values():
            assert p.is_file()
        img = Image.open(out["overlay"])
        assert img.size == (base.shape[1], base.shape[0])
        html_text = out["report"].read_text(encoding="utf-8")
        assert regions_payload["image_sha256"] in html_text  # SHA-256 绑定展示
        assert json.dumps(cs.to_params(), ensure_ascii=False) in html_text or "ember" in html_text

    def test_requires_sha_binding(self, analysis, base, density, tmp_path):
        cs = ColorScale("ember", 0.0, 1.0)
        with pytest.raises(ValueError, match="SHA-256"):
            render_run_report(
                out_dir=tmp_path / "run",
                base_image=base,
                density=density,
                analysis={"input": {}},
                regions_payload={"regions": []},
                colorscale=cs,
            )


# ---------------------------------------------------------------------------
# compare 视图
# ---------------------------------------------------------------------------

def _side_regions() -> list[dict]:
    return [
        {
            "id": "claim-button",
            "label": XSS_LABEL,
            "role": None,
            "geometry": {"type": "rect", "x": 10, "y": 8, "width": 20, "height": 6},
            "source": "manual",
            "status": "confirmed",
        }
    ]


def _comparison_fixture(compatible: bool = True) -> dict:
    return {
        "schema_version": "game-ui-attention-comparison/v1",
        "before": {"analysis_id": "an-A"},
        "after": {"analysis_id": "an-B"},
        "compatibility": {
            "compatible": compatible,
            "mismatches": []
            if compatible
            else [{"field": "model.version", "description": "后端实现版本", "before": "0.2.0", "after": "0.3.0"}],
            "canvas_note": "画布尺寸相同，满足第一版正式比较条件",
            "exploratory": False,
            "exploratory_reasons": [],
        },
        "regions": {
            "matched": [
                {
                    "id": "claim-button",
                    "label": XSS_LABEL,
                    "mass_before": 0.30,
                    "mass_after": 0.4234567,
                    "delta_pp": 12.34567,
                    "area_px_before": 120,
                    "area_px_after": 150,
                    "area_px_change": 30,
                    "relative_density_before": 7.68,
                    "relative_density_after": 8.9,
                    "geometry_before": {"type": "rect", "x": 10, "y": 8, "width": 20, "height": 6},
                    "geometry_after": {"type": "rect", "x": 10, "y": 8, "width": 25, "height": 6},
                }
            ],
            "added": [
                {
                    "id": "new-banner",
                    "label": "新横幅 & 活动",
                    "role": None,
                    "source": "manual",
                    "status": "candidate",
                    "geometry": {"type": "rect", "x": 2, "y": 2, "width": 30, "height": 5},
                    "area_px": 150,
                    "area_fraction": 0.048,
                    "probability_mass": 0.09,
                    "relative_density": 1.87,
                }
            ],
            "removed": [
                {
                    "id": "old-popup",
                    "label": "旧弹窗",
                    "role": None,
                    "source": "agent",
                    "status": "confirmed",
                    "geometry": {"type": "rect", "x": 5, "y": 5, "width": 10, "height": 10},
                    "area_px": 100,
                    "area_fraction": 0.032,
                    "probability_mass": 0.05,
                    "relative_density": 1.5,
                }
            ],
        },
        "limitations": ["delta_pp 只说明模型预测分布发生变化"],
    }


class TestCompareView:
    def _render(self, tmp_path, base, density, compatible=True):
        comparison = _comparison_fixture(compatible)
        out = render_compare_report(
            out_dir=tmp_path / "cmp",
            comparison=comparison,
            base_a=base,
            density_a=density,
            regions_a=_side_regions(),
            base_b=base,
            density_b=make_density(peak=0.03),
            regions_b=_side_regions(),
        )
        return out, out["report"].read_text(encoding="utf-8")

    def test_shared_colorscale_single_param_set(self, tmp_path, base, density):
        import html as html_mod

        out, html_text = self._render(tmp_path, base, density)
        # 两侧 overlay 使用同一份色阶参数：参数 JSON 在页面出现且两侧标注一致
        # （页面内 JSON 文本经 HTML 转义，解析前先 unescape——转义本身是被测行为）
        params = json.loads(
            html_mod.unescape(re.search(r"参数随 manifest 记录：<code>(.*?)</code>", html_text).group(1))
        )
        assert set(params["colorscale"]) == {"name", "vmin", "vmax", "normalization"}
        side_notes = [
            html_mod.unescape(m)
            for m in re.findall(r"叠加图使用 A/B 共用色阶\s*<code>(.*?)</code>", html_text)
        ]
        assert len(side_notes) == 2
        assert side_notes[0] == side_notes[1]
        assert json.loads(side_notes[0]) == params["colorscale"]
        # 共用值域覆盖两侧（vmax 来自 B 侧更大密度）
        assert params["colorscale"]["vmax"] > params["colorscale"]["vmin"]
        assert out["overlay_a"].is_file() and out["overlay_b"].is_file()

    def test_delta_table_and_single_side_sections(self, tmp_path, base, density):
        _, html_text = self._render(tmp_path, base, density)
        assert "配对区域差值" in html_text
        assert "12.3457" in html_text  # delta_pp 显示舍入（原样数值）
        assert "0.423457" in html_text  # mass_after
        assert "新增区域（仅 B 侧存在）" in html_text
        assert "移除区域（仅 A 侧存在）" in html_text
        assert "new-banner" in html_text and "old-popup" in html_text
        assert "不编造缺失一侧的 0 值" in html_text

    def test_compare_xss_escaped(self, tmp_path, base, density):
        _, html_text = self._render(tmp_path, base, density)
        assert "<script>alert" not in html_text
        assert "&lt;script&gt;alert(&#x27;xss-label&#x27;)" in html_text
        assert "新横幅 &amp; 活动" in html_text
        assert html_text.count("<script") == 2  # 数据岛 + 内联 js
        island = split_island(html_text)[1]
        assert "</script" not in island

    def test_incompatible_hides_delta_values(self, tmp_path, base, density):
        _, html_text = self._render(tmp_path, base, density, compatible=False)
        assert "拒绝正式比较" in html_text
        assert "COMPARISON_INCOMPATIBLE" in html_text
        assert "model.version" in html_text  # 保留原因
        assert "12.3457" not in html_text  # 不展示差值数值

    def test_compare_no_external_links_and_csp(self, tmp_path, base, density):
        _, html_text = self._render(tmp_path, base, density)
        lowered = html_text.lower()
        assert "http://" not in lowered and "https://" not in lowered
        assert "<script src" not in lowered
        policy = re.search(r'Content-Security-Policy" content="([^"]+)"', html_text).group(1)
        assert "default-src 'none'" in policy
        script_body = re.search(r"<script>(.*?)</script>", html_text, re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(script_body.encode("utf-8")).digest()).decode()
        assert f"'sha256-{digest}'" in policy

    def test_low_level_entry_accepts_precomputed_sides(self, tmp_path, base, density):
        comparison = _comparison_fixture()
        side = {
            "analysis_id": "an-X",
            "base_png_bytes": png_bytes(base),
            "overlay_png_bytes": png_bytes(base),
            "regions": _side_regions(),
            "image_size": (base.shape[1], base.shape[0]),
        }
        cs = ColorScale.shared("ember", density, make_density(peak=0.03))
        out = render_comparison_html(
            comparison=comparison,
            side_a=side,
            side_b=dict(side, analysis_id="an-Y"),
            colorscale_params=cs.to_params(),
            overlay_alpha=0.55,
            out_path=tmp_path / "compare.html",
        )
        text = out.read_text(encoding="utf-8")
        assert "an-X" in text and "an-Y" in text
        assert out.is_file()
