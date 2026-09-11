"""ui_attention.report — 本地报告渲染层（C3 独占目录）。

模块一览：

- :mod:`colorscale`：原创色表 LUT、可序列化色阶参数、A/B 共用值域
  ``shared_range``、确定性密度→RGBA 映射。
- :mod:`overlay`：原图+热图叠加 PNG（固定透明度、可记录、尺寸与原图一致、
  确定性输出）与纯热图 RGBA PNG。
- :mod:`export_regions`：独立 regions JSON 导出（图片 SHA-256 强制绑定）
  与校验纯函数（拒绝越界/零面积/自交/重复 ID/错图绑定；C1 冻结契约兜底）。
- :mod:`report`：自包含 report.html（查看切换/区域列表/圈选编辑导出/
  指标表/评审区；全转义、CSP 内容哈希收紧、无外链、无自动网络请求）。
- :mod:`compare_view`：A/B 并排 + 共用色阶渲染 + delta_pp 表 +
  新增/移除区域单列区。

全部渲染入口为纯函数式（输入数据 + 配置 → 输出文件路径），供 C1
cli.py 接线；签名详见各模块 docstring。渲染层不重新计算任何统计、
不从 PNG 颜色反推数值（technical-design §9）。
"""

from .colorscale import (
    LUT_SIZE,
    NORMALIZATIONS,
    ColorScale,
    build_lut,
    color_table_names,
    shared_range,
)
from .compare_view import (
    SharedCompareRender,
    colorbar_png_bytes,
    render_compare_report,
    render_comparison_html,
    render_shared_overlays,
)
from .export_regions import (
    REGION_SOURCES,
    REGION_STATUSES,
    REGIONS_SCHEMA_VERSION,
    RegionsValidationResult,
    compute_image_sha256,
    export_regions_file,
    load_and_validate_regions_file,
    make_regions_payload,
    normalize_region_for_export,
    validate_regions_payload,
)
from .overlay import (
    DEFAULT_OVERLAY_ALPHA,
    overlay_record,
    overlay_rgba,
    render_heatmap_png,
    render_overlay_png,
    resize_density_nearest,
)
from .report import build_report_html, render_report_html, render_run_report

__all__ = [
    "DEFAULT_OVERLAY_ALPHA",
    "LUT_SIZE",
    "NORMALIZATIONS",
    "REGIONS_SCHEMA_VERSION",
    "REGION_SOURCES",
    "REGION_STATUSES",
    "ColorScale",
    "RegionsValidationResult",
    "SharedCompareRender",
    "build_lut",
    "build_report_html",
    "color_table_names",
    "colorbar_png_bytes",
    "compute_image_sha256",
    "export_regions_file",
    "load_and_validate_regions_file",
    "make_regions_payload",
    "normalize_region_for_export",
    "overlay_record",
    "overlay_rgba",
    "render_compare_report",
    "render_comparison_html",
    "render_heatmap_png",
    "render_overlay_png",
    "render_report_html",
    "render_run_report",
    "render_shared_overlays",
    "resize_density_nearest",
    "shared_range",
    "validate_regions_payload",
]
