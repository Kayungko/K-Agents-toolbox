"""CLI 入口：doctor / analyze / summarize / compare（data-contract.md §1/§5/§7）。

约定：
- stdout 只输出一个 JSON 信封（ok/result/error），进度走 stderr；调用方按退出码分支；
- 运行产物写入 runs/ 下**新目录**：目录已存在 → 退出码 7，不覆盖历史；
- manifest.json **最后落盘**表示计算完成；失败路径不写 manifest、清理自建的部分产物（无假成功）；
- doctor/analyze 经 ``backends/registry``（C2）动态导入解析 profile；后端缺失 → 退出码 3 结构化报错，不崩溃；
- summarize 只读 density.npy 重算区域统计（校验图片 SHA-256），**不重新推理**；
- compare 做兼容性校验 + 稳定 ID 配对 + delta_pp；不兼容 → 退出码 6 并保留原因；
- overlay.png / heatmap.png / report.html 由 C3 report 模块渲染（动态导入
  ``report.render_run_report`` / ``compare_view.render_compare_report``，总控接线点①）；
  共享色阶与 overlay 参数写入 manifest（``colorscale``/``overlay`` 键）；
  模块缺失或渲染失败时计算结果仍有效（部分完成），缺失产物不列为成功。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, imaging
from .aoi import build_region_masks
from .contracts import (
    AnalysisRecord,
    ArtifactRecord,
    RegionsFile,
    build_comparison,
    load_analysis,
    load_analyze_request,
    load_regions_file,
    load_review,
)
from .contracts.analysis import ANALYSIS_SCHEMA_VERSION, METRICS_VERSION
from .errors import (
    EXIT_INCOMPATIBLE,
    EXIT_INTERNAL,
    EXIT_INVALID_INPUT,
    EXIT_INVALID_OUTPUT,
    EXIT_IO,
    EXIT_NOT_READY,
    EXIT_OK,
    EXIT_RUNTIME,
    ErrorCode,
    UiAttentionError,
    emit_error,
    emit_success,
    ensure_utf8_streams,
    progress,
)
from .metrics import (
    log_density_to_probability,
    region_statistics,
    resample_to_original,
    union_statistics,
    validate_probability,
)

#: C2 registry 模块路径（动态导入面；缺失 → MODEL_NOT_READY 退出码 3）
REGISTRY_MODULE = "ui_attention.backends.registry"
#: C3 report 模块路径（动态导入面；缺失 → 计算仍成功，报告产物记为缺失）
REPORT_MODULE = "ui_attention.report"

MANIFEST_SCHEMA_VERSION = "game-ui-attention-manifest/v1"

_DEPENDENCY_PACKAGES = ("numpy", "pillow", "scipy", "onnxruntime")


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_analysis_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"analysis-{stamp}-{uuid.uuid4().hex[:8]}"


def prepare_run_directory(out: str | Path) -> Path:
    """运行目录必须不存在（不覆盖历史报告）；已存在 → OUTPUT_PATH_EXISTS 退出码 7。"""
    p = Path(out)
    if p.exists():
        raise UiAttentionError(
            ErrorCode.OUTPUT_PATH_EXISTS,
            f"输出路径已存在，拒绝覆盖历史结果：{p}",
            {"path": str(p)},
        )
    try:
        p.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise UiAttentionError(ErrorCode.IO_ERROR, f"输出目录创建失败：{p}", {"reason": str(exc)}) from exc
    return p


def _write_json(path: Path, obj: Any) -> None:
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError as exc:
        raise UiAttentionError(ErrorCode.IO_ERROR, f"写入失败：{path.name}", {"reason": str(exc)}) from exc


def _artifact_record(run_dir: Path, rel_path: str) -> ArtifactRecord:
    p = run_dir / rel_path
    if not p.is_file():
        raise UiAttentionError(
            ErrorCode.ARTIFACT_CHECK_FAILED,
            f"产物缺失，不列为成功：{rel_path}",
        )
    return ArtifactRecord(path=rel_path, sha256=imaging.sha256_file(p), size_bytes=p.stat().st_size)


def _write_manifest(
    run_dir: Path,
    *,
    command: str,
    computation_status: str,
    analysis_id: str | None,
    extra: dict[str, Any] | None = None,
) -> ArtifactRecord:
    """manifest.json 最后落盘，表示计算完成（technical-design §10）。"""
    artifacts = []
    for name in sorted(p.name for p in run_dir.iterdir() if p.is_file()):
        if name == "manifest.json":
            continue
        artifacts.append(_artifact_record(run_dir, name).to_dict())
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utcnow(),
        "tool_version": __version__,
        "command": command,
        "computation_status": computation_status,
        "analysis_id": analysis_id,
        "artifacts": artifacts,
        "note": "manifest 最后落盘表示计算完成；计算完成不代表评审完成，也不代表模型预测准确",
    }
    if extra:
        manifest.update(extra)
    _write_json(run_dir / "manifest.json", manifest)
    return ArtifactRecord(
        path="manifest.json",
        sha256=imaging.sha256_file(run_dir / "manifest.json"),
        size_bytes=(run_dir / "manifest.json").stat().st_size,
    )


def _collect_dependencies() -> dict[str, str]:
    deps = {"python": platform.python_version(), "ui_attention": __version__}
    for pkg in _DEPENDENCY_PACKAGES:
        try:
            deps[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            deps[pkg] = "not-installed"
    return deps


def import_registry() -> Any:
    """动态导入 C2 registry；模块缺失 → MODEL_NOT_READY（退出码 3），不崩溃。"""
    try:
        return importlib.import_module(REGISTRY_MODULE)
    except ImportError as exc:
        raise UiAttentionError(
            ErrorCode.MODEL_NOT_READY,
            f"后端 registry 不可用（{REGISTRY_MODULE}）：后端未实现或依赖未安装；"
            "模型下载是独立的显式安装步骤，不静默替代",
            {"module": REGISTRY_MODULE, "reason": str(exc)},
        ) from exc


def _import_report_module() -> tuple[Any | None, str | None]:
    """动态导入 C3 report 模块；返回 (module|None, 导入失败原因|None)。

    模块缺失/损坏都不得使计算失败（technical-design §10：报告失败保留有效计算结果）。
    """
    try:
        return importlib.import_module(REPORT_MODULE), None
    except Exception as exc:  # noqa: BLE001 - report 模块缺失/损坏都不得使计算失败
        if isinstance(exc, ImportError):
            return None, None
        return None, f"report 模块导入失败（{type(exc).__name__}），报告产物未生成（计算结果保留为部分完成）"


def _render_run_report(
    run_dir: Path,
    *,
    base_image: np.ndarray,
    density: np.ndarray,
    analysis_dict: dict[str, Any],
    regions_payload: dict[str, Any],
    review: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """单图渲染（analyze/summarize 共用；总控接线点①）。

    调 C3 ``report.render_run_report`` → overlay.png / heatmap.png / report.html；
    manifest 记录 ``overlay``（overlay_record）与 ``colorscale``（色阶参数）。
    单图色阶值域由该图密度经 ``ColorScale.shared(name, d, d)`` 确定性生成；
    A/B 共用色阶在 :func:`_render_compare_report` 中以联合值域生成（不各自拉满）。
    返回 ``{"rendered": [文件名], "overlay": dict|None, "colorscale": dict|None, "error": str|None}``。
    """
    out: dict[str, Any] = {"rendered": [], "overlay": None, "colorscale": None, "error": None}
    module, err = _import_report_module()
    if module is None:
        out["error"] = err
        return out
    render = getattr(module, "render_run_report", None)
    if render is None:
        return out
    try:
        colorscale = module.ColorScale.shared("ember", density, density)
        alpha = module.DEFAULT_OVERLAY_ALPHA
        base_png = run_dir / "base.png"
        paths = render(
            out_dir=run_dir,
            base_image=base_image,
            density=density,
            analysis=analysis_dict,
            regions_payload=regions_payload,
            colorscale=colorscale,
            alpha=alpha,
            review=review,
            base_png_path=base_png if base_png.is_file() else None,
            title=title,
        )
        out["rendered"] = [Path(p).name for p in (paths or {}).values()]
        out["overlay"] = module.overlay_record(colorscale, alpha)
        out["colorscale"] = colorscale.to_params()
    except Exception as exc:  # noqa: BLE001 - 渲染失败保留有效计算结果，报告为部分完成
        out["error"] = f"report 渲染失败（render_run_report）：{type(exc).__name__}: {exc}"
    return out


def _render_compare_report(
    run_dir: Path,
    *,
    comparison: dict[str, Any],
    base_a: np.ndarray,
    density_a: np.ndarray,
    regions_a: list[dict[str, Any]],
    base_b: np.ndarray,
    density_b: np.ndarray,
    regions_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """A/B 对比渲染（总控接线点①③：density 必须已是各自原图尺寸）。

    C3 ``compare_view.render_compare_report`` 内部经 ``ColorScale.shared`` 以
    A/B 联合值域生成同一色阶实例作用两侧（validation-plan §1 色阶行锚点）；
    manifest 的 colorscale/overlay 记录由本地以相同输入确定性重建（同一函数、
    同一默认参数 → 与渲染所用参数一致）。
    """
    out: dict[str, Any] = {"rendered": [], "overlay": None, "colorscale": None, "error": None}
    module, err = _import_report_module()
    if module is None:
        out["error"] = err
        return out
    render = getattr(module, "render_compare_report", None)
    if render is None:
        return out
    try:
        alpha = module.DEFAULT_OVERLAY_ALPHA
        paths = render(
            out_dir=run_dir,
            comparison=comparison,
            base_a=base_a,
            density_a=density_a,
            regions_a=regions_a,
            base_b=base_b,
            density_b=density_b,
            regions_b=regions_b,
            alpha=alpha,
        )
        shared = module.ColorScale.shared("ember", density_a, density_b)
        out["rendered"] = [Path(p).name for p in (paths or {}).values()]
        out["overlay"] = module.overlay_record(shared, alpha)
        out["colorscale"] = shared.to_params()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"A/B 报告渲染失败（render_compare_report）：{type(exc).__name__}: {exc}"
    return out


def _c3_validate_regions_file(path: str | Path, expected_image_sha256: str, image_size_wh: tuple[int, int]) -> None:
    """C3 兜底校验（总控接线点④）：与 contracts/regions.py 权威校验互为兜底。

    C3 校验器不可用时静默跳过（权威校验已通过）；校验不通过 → INVALID_AOI（退出码 2）。
    """
    module, _err = _import_report_module()
    if module is None:
        return
    fn = getattr(module, "load_and_validate_regions_file", None)
    if fn is None:
        return
    try:
        _payload, result = fn(Path(path), expected_image_sha256=expected_image_sha256, image_size=image_size_wh)
    except Exception:  # noqa: BLE001 - 兜底校验器自身异常不阻断主链
        return
    if result is not None and not getattr(result, "ok", True):
        raise UiAttentionError(
            ErrorCode.INVALID_AOI,
            "regions 文件被 C3 兜底校验拒绝（错图绑定/越界/零面积/自交/重复 ID 等）",
            {"errors": [dict(e) for e in getattr(result, "errors", ())]},
        )


def _region_spec_rows(region_result_rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """analysis.regions 结果行 → RegionSpec.to_dict 形态（C3 区域框静态渲染输入）。"""
    rows = []
    for r in region_result_rows:
        rows.append(
            {
                "id": r.get("id"),
                "label": r.get("label"),
                "role": r.get("role"),
                "geometry": r.get("geometry"),
                "source": r.get("source"),
                "status": r.get("status"),
            }
        )
    return rows


def _load_compare_materials(
    before_dir: Path,
    before_rec: AnalysisRecord,
    after_dir: Path,
    after_rec: AnalysisRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """读取两侧运行目录的渲染素材 (base_a, density_a, base_b, density_b)。

    任一缺失/尺寸不符/数值异常 → None（跳过渲染并记录原因，comparison.json 仍为正式产物）。
    density.npy 由 analyze/summarize 落盘，已是原图尺寸（总控接线点③：
    统计路径的重采样在 metrics 完成，渲染只消费原图尺寸密度）。
    """
    try:
        out: list[np.ndarray] = []
        for run, rec in ((before_dir, before_rec), (after_dir, after_rec)):
            base_p = run / "base.png"
            den_p = run / "density.npy"
            if not (base_p.is_file() and den_p.is_file()):
                return None
            base = imaging.load_rgb_png(base_p)
            density = np.asarray(np.load(den_p), dtype=np.float64)
            h, w = int(rec.input["height"]), int(rec.input["width"])
            if base.shape[:2] != (h, w) or density.shape != (h, w):
                return None
            if not np.all(np.isfinite(density)) or np.any(density < 0):
                return None
            out.extend([base, density])
        return (out[0], out[1], out[2], out[3])
    except Exception:  # noqa: BLE001 - 素材读取的任何异常都只跳过渲染，不影响比较结果
        return None


def _parse_background(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        raise UiAttentionError(
            ErrorCode.INVALID_IMAGE,
            f"--composite-background 必须是 'R,G,B'（0-255），得到 {value!r}",
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _cleanup_partial(run_dir: Path) -> None:
    """失败恢复：清理本次自建的运行目录（不产生假成功、不留半成品；不动历史目录）。"""
    shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    progress("doctor：检查后端 registry、依赖、权重与许可记录…")
    registry = import_registry()  # 缺失 → MODEL_NOT_READY（退出码 3）
    for fn in ("doctor_report", "list_profiles"):
        if not hasattr(registry, fn):
            raise UiAttentionError(
                ErrorCode.MODEL_NOT_READY,
                f"registry 模块缺少期望接口 {fn}()（BackendRegistry Protocol 未满足）",
                {"module": REGISTRY_MODULE, "missing": fn},
            )
    report = registry.doctor_report()
    emit_success(
        {
            "tool_version": __version__,
            "dependencies": _collect_dependencies(),
            "profiles": list(registry.list_profiles()),
            "doctor": report,
        }
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _prediction_to_probability(result: Any, info: Any) -> np.ndarray:
    """按冻结 semantics 把 PredictionResult.array 转为推理网格上的严格概率图。"""
    if "spatial_density" not in tuple(info.capabilities):
        raise UiAttentionError(
            ErrorCode.UNSUPPORTED_SEMANTICS,
            "后端未声明 spatial_density capability：第一版统计管线只消费空间密度",
            {"capabilities": list(info.capabilities)},
        )
    if result.semantics == "log_density":
        progress("semantics=log_density：P = exp(L − logsumexp(L))")
        return log_density_to_probability(result.array)
    if result.semantics == "probability_density":
        # 适配层已完成 sum=1 重归一化后声明 probability_density：严格校验，不静默修补
        validate_probability(np.asarray(result.array, dtype=np.float64))
        return np.asarray(result.array, dtype=np.float64)
    raise UiAttentionError(
        ErrorCode.UNSUPPORTED_SEMANTICS,
        f"后端输出语义 {result.semantics!r} 不被第一版消费（厂商分数不反推像素概率、不冒充本方案指标）",
        {"semantics": result.semantics},
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    request = load_analyze_request(args.request)
    background = _parse_background(args.composite_background)
    run_dir = prepare_run_directory(args.out)
    try:
        progress(f"读取并校验图片：{request.image}")
        image = imaging.load_image(request.image_path, composite_background=background)
        request.check_bounds(image.width, image.height)

        progress(f"解析 backend_profile：{request.backend_profile}")
        registry = import_registry()
        profile = registry.resolve_profile(request.backend_profile)
        backend = registry.get_backend(profile)
        info = backend.describe()
        info.validate()

        progress("推理中（模型侧预处理由后端按 profile 内部执行）…")
        result = backend.predict(image.array, profile)
        result.validate()

        p_inference = _prediction_to_probability(result, info)
        progress(f"映射回原图尺寸 {image.width}×{image.height} 并重归一化…")
        density = resample_to_original(p_inference, result.shape_mapping)

        progress("AOI 掩码与区域统计…")
        masks = build_region_masks(request.regions, (image.height, image.width))
        region_rows: list[dict[str, Any]] = []
        for region in request.regions:
            stats = region_statistics(density, masks[region.id])
            region_rows.append(
                {
                    "id": region.id,
                    "label": region.label,
                    "role": region.role,
                    "source": region.source,
                    "status": region.status,
                    "geometry": region.geometry.to_dict(),
                    "area_px": stats["area_px"],
                    "area_fraction": stats["area_fraction"],
                    "probability_mass": stats["probability_mass"],
                    "relative_density": stats["relative_density"],
                }
            )
        region_union = None
        if masks:
            union_stats = union_statistics(density, list(masks.values()))
            region_union = {
                "area_px": union_stats["area_px"],
                "area_fraction": union_stats["area_fraction"],
                "probability_mass": union_stats["probability_mass"],
                "relative_density": union_stats["relative_density"],
                "overlap_dedup_px": union_stats["overlap_dedup_px"],
                "note": "汇总按像素掩码并集去重；指标不可简单求和",
            }

        limitations = list(result.limitations)
        if request.player_goal is None:
            limitations.append("缺少玩家目标：不得声称目标完成路径合理，仅描述性统计")
        candidate_regions = [r["id"] for r in region_rows if r["status"] == "candidate"]
        if candidate_regions:
            limitations.append(f"候选边界区域（指标同样标为基于候选边界）：{candidate_regions}")
        if image.alpha_composited:
            limitations.append(f"图片带透明通道，已按显式合成背景 {image.composite_background} 合成")

        # 产物：density.npy（全精度 float64）→ regions.json → analysis.json → (C3 报告) → manifest 最后
        progress("落盘 density.npy / regions.json / analysis.json…")
        np.save(run_dir / "density.npy", density)
        regions_out = RegionsFile(image_sha256=image.sha256, regions=request.regions)
        _write_json(run_dir / "regions.json", regions_out.to_dict())

        analysis_obj = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_id": _new_analysis_id(),
            "computation_status": "complete",
            "review_status": "not_requested",
            "evidence_type": "model_prediction",
            "created_at_utc": _utcnow(),
            "metrics_version": METRICS_VERSION,
            "player_goal": request.player_goal,
            "input": {
                "image_path": request.image,  # 保留请求原样（相对路径），不落盘机器绝对路径
                "image_sha256": image.sha256,
                "width": image.width,
                "height": image.height,
                "inference_width": int(result.shape_mapping["inference_shape"][1]),
                "inference_height": int(result.shape_mapping["inference_shape"][0]),
                "screen_type": request.screen_type,
            },
            "model": {
                "backend_id": info.backend_id,
                "version": info.version,
                "weights_sha256": [w.sha256 for w in info.weights],
                "code_version": info.version,
                "capabilities": list(info.capabilities),
                "native_semantics": info.native_semantics,
                "license_status": dict(info.license_status),
            },
            "profile": profile.to_dict(),
            "runtime": {
                "dependencies": _collect_dependencies(),
                "device": result.runtime["device"],
                "precision": result.runtime["precision"],
                "elapsed_ms": result.runtime["elapsed_ms"],
                "peak_mem_mb": result.runtime["peak_mem_mb"],
            },
            "regions": region_rows,
            "region_union": region_union,
            "limitations": limitations,
            "artifacts": [],
            "errors": [],
        }
        # profile 记录必须含实际执行解析结果，且满足 analysis v1 的 profile 键集合
        analysis_obj["profile"] = {
            "profile_name": profile.profile_name,
            "config_hash": profile.config_hash,
            "preprocessing": dict(profile.preprocessing),
            "centerbias": dict(profile.centerbias) if profile.centerbias is not None else None,
            "viewing_conditions": dict(profile.viewing_conditions),
        }
        analysis_obj["artifacts"] = [
            _artifact_record(run_dir, "density.npy").to_dict(),
            _artifact_record(run_dir, "regions.json").to_dict(),
        ]
        record = AnalysisRecord.from_dict(analysis_obj)  # 自校验（无效输出不产生假成功）
        _write_json(run_dir / "analysis.json", record.to_dict())

        # base.png：方向已处理原图的展示副本（C3 渲染内嵌；runs/ 被 gitignore 覆盖；
        # 非哈希绑定原文件——图片哈希绑定以 analysis.input.image_sha256 为准）
        imaging.save_rgb_png(image.array, run_dir / "base.png")

        render = _render_run_report(
            run_dir,
            base_image=image.array,
            density=density,
            analysis_dict=record.to_dict(),
            regions_payload=regions_out.to_dict(),
        )
        if render["error"]:
            # 报告部分完成：保留有效计算结果（technical-design §10）
            analysis_obj["limitations"].append(render["error"])
            record = AnalysisRecord.from_dict(analysis_obj)
            _write_json(run_dir / "analysis.json", record.to_dict())
        if not render["rendered"]:
            progress("overlay/report 未生成：report 模块不可用或渲染失败（计算结果仍有效，缺失产物不列为成功）")

        manifest_extra: dict[str, Any] = {"rendered_report_artifacts": render["rendered"]}
        if render["overlay"] is not None:
            manifest_extra["overlay"] = render["overlay"]
        if render["colorscale"] is not None:
            manifest_extra["colorscale"] = render["colorscale"]
        manifest = _write_manifest(
            run_dir,
            command="analyze",
            computation_status="complete",
            analysis_id=record.analysis_id,
            extra=manifest_extra,
        )
        emit_success(
            {
                "run_directory": str(args.out),
                "analysis_id": record.analysis_id,
                "computation_status": "complete",
                "image_sha256": image.sha256,
                "backend": {"backend_id": info.backend_id, "version": info.version, "profile": profile.profile_name},
                "artifacts": [a.to_dict() for a in record.artifacts] + [manifest.to_dict()],
                "regions": len(region_rows),
                "limitations": record.limitations,
            }
        )
        return EXIT_OK
    except UiAttentionError:
        _cleanup_partial(run_dir)  # 失败恢复：无假成功、无半成品目录
        raise


# ---------------------------------------------------------------------------
# summarize（只读 density.npy 重算区域，不重新推理）
# ---------------------------------------------------------------------------


def cmd_summarize(args: argparse.Namespace) -> int:
    src_run = Path(args.analysis)
    prior = load_analysis(src_run)  # 不存在/损坏 → ARTIFACT_CHECK_FAILED（退出码 5）
    density_path = src_run / "density.npy"
    if not density_path.is_file():
        raise UiAttentionError(
            ErrorCode.ARTIFACT_CHECK_FAILED,
            f"源运行目录缺少 density.npy，无法只读重算：{src_run.name}",
        )
    density_sha = imaging.sha256_file(density_path)
    recorded = next((a for a in prior.artifacts if Path(a.path).name == "density.npy"), None)
    if recorded is not None and recorded.sha256 != density_sha:
        raise UiAttentionError(
            ErrorCode.ARTIFACT_CHECK_FAILED,
            "density.npy 哈希与 analysis 记录不符（产物被改动或损坏），拒绝重算",
            {"recorded": recorded.sha256, "actual": density_sha},
        )
    try:
        density = np.load(density_path)
    except (OSError, ValueError) as exc:
        raise UiAttentionError(ErrorCode.ARTIFACT_CHECK_FAILED, "density.npy 读取失败", {"reason": str(exc)}) from exc
    validate_probability(np.asarray(density, dtype=np.float64))
    density = np.asarray(density, dtype=np.float64)

    regions_file = load_regions_file(args.regions)
    # 图片 SHA-256 绑定校验：防止把标注应用到其他图片（不重新推理）
    regions_file.check_image_sha256(prior.input["image_sha256"])
    width, height = int(prior.input["width"]), int(prior.input["height"])
    if density.shape != (height, width):
        raise UiAttentionError(
            ErrorCode.ARTIFACT_CHECK_FAILED,
            f"density.npy 形状 {density.shape} 与原图尺寸 {(height, width)} 不符",
        )
    regions_file.check_bounds(width, height)
    # C3 兜底校验（总控接线点④）：HTML 导出 regions 的错图绑定/几何错误双重防线
    _c3_validate_regions_file(args.regions, str(prior.input["image_sha256"]), (width, height))

    run_dir = prepare_run_directory(args.out)
    try:
        progress("按新标注重算区域统计（复用既有概率图，不重新推理）…")
        masks = build_region_masks(regions_file.regions, (height, width))
        region_rows = []
        for region in regions_file.regions:
            stats = region_statistics(density, masks[region.id])
            region_rows.append(
                {
                    "id": region.id,
                    "label": region.label,
                    "role": region.role,
                    "source": region.source,
                    "status": region.status,
                    "geometry": region.geometry.to_dict(),
                    "area_px": stats["area_px"],
                    "area_fraction": stats["area_fraction"],
                    "probability_mass": stats["probability_mass"],
                    "relative_density": stats["relative_density"],
                }
            )
        region_union = None
        if masks:
            union_stats = union_statistics(density, list(masks.values()))
            region_union = {
                "area_px": union_stats["area_px"],
                "area_fraction": union_stats["area_fraction"],
                "probability_mass": union_stats["probability_mass"],
                "relative_density": union_stats["relative_density"],
                "overlap_dedup_px": union_stats["overlap_dedup_px"],
                "note": "汇总按像素掩码并集去重；指标不可简单求和",
            }

        shutil.copyfile(density_path, run_dir / "density.npy")
        copied_sha = imaging.sha256_file(run_dir / "density.npy")
        if copied_sha != density_sha:
            raise UiAttentionError(ErrorCode.ARTIFACT_CHECK_FAILED, "density.npy 复制后哈希不一致（IO 损坏）")

        _write_json(run_dir / "regions.json", regions_file.to_dict())

        # 底图与评审：源运行目录存在则复用（展示副本 + Agent 评审只校验与呈现，不生成数值）
        base_image: np.ndarray | None = None
        src_base = src_run / "base.png"
        if src_base.is_file():
            base_image = imaging.load_rgb_png(src_base)
            if base_image.shape[:2] != (height, width):
                raise UiAttentionError(
                    ErrorCode.ARTIFACT_CHECK_FAILED,
                    f"源运行 base.png 尺寸 {base_image.shape[:2]} 与原图 {(height, width)} 不符",
                )
            shutil.copyfile(src_base, run_dir / "base.png")
        review_dict: dict[str, Any] | None = None
        src_review = src_run / "review.json"
        if src_review.is_file():
            review_dict = load_review(src_review).to_dict()  # 无效 review → 退出码 2
            shutil.copyfile(src_review, run_dir / "review.json")

        limitations = list(prior.limitations) + [
            "summarize：仅按新标注重算区域统计，未重新推理；底层概率图哈希保持不变",
        ]
        if base_image is None:
            limitations.append("源运行目录无 base.png：overlay/report.html 未重生成（区域统计仍有效）")
        analysis_obj = prior.to_dict()
        analysis_obj.update(
            {
                "analysis_id": _new_analysis_id(),
                "created_at_utc": _utcnow(),
                "computation_status": "complete",
                "regions": region_rows,
                "region_union": region_union,
                "limitations": limitations,
            }
        )
        analysis_obj["artifacts"] = [
            _artifact_record(run_dir, "density.npy").to_dict(),
            _artifact_record(run_dir, "regions.json").to_dict(),
        ]
        record = AnalysisRecord.from_dict(analysis_obj)
        _write_json(run_dir / "analysis.json", record.to_dict())

        render: dict[str, Any] = {"rendered": [], "overlay": None, "colorscale": None, "error": None}
        if base_image is not None:
            render = _render_run_report(
                run_dir,
                base_image=base_image,
                density=density,
                analysis_dict=record.to_dict(),
                regions_payload=regions_file.to_dict(),
                review=review_dict,
            )
            if render["error"]:
                analysis_obj["limitations"].append(render["error"])
                record = AnalysisRecord.from_dict(analysis_obj)
                _write_json(run_dir / "analysis.json", record.to_dict())
        manifest_extra = {
            "source_analysis_id": prior.analysis_id,
            "density_sha256": density_sha,
            "rendered_report_artifacts": render["rendered"],
        }
        if render["overlay"] is not None:
            manifest_extra["overlay"] = render["overlay"]
        if render["colorscale"] is not None:
            manifest_extra["colorscale"] = render["colorscale"]
        manifest = _write_manifest(
            run_dir,
            command="summarize",
            computation_status="complete",
            analysis_id=record.analysis_id,
            extra=manifest_extra,
        )
        emit_success(
            {
                "run_directory": str(args.out),
                "analysis_id": record.analysis_id,
                "source_analysis_id": prior.analysis_id,
                "density_sha256": density_sha,  # 相同底层概率图哈希（标注变更不重推理）
                "recomputed_regions": len(region_rows),
                "artifacts": [a.to_dict() for a in record.artifacts] + [manifest.to_dict()],
            }
        )
        return EXIT_OK
    except UiAttentionError:
        _cleanup_partial(run_dir)
        raise


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    before = load_analysis(args.before)
    after = load_analysis(args.after)
    run_dir = prepare_run_directory(args.out)
    try:
        progress("A/B 兼容性校验（模型/权重/预处理/先验/画布尺寸/指标版本）…")
        comparison = build_comparison(before, after, created_at_utc=_utcnow())  # 不兼容 → 退出码 6，原因保留
        obj = comparison.to_dict()
        obj["before"]["run_directory"] = Path(args.before).name
        obj["after"]["run_directory"] = Path(args.after).name
        _write_json(run_dir / "comparison.json", obj)

        # A/B 渲染素材：两侧运行目录的 base.png 与 density.npy（density 已是原图尺寸；
        # 缺失 → 跳过渲染并记录，comparison.json 仍为正式产物）
        render: dict[str, Any] = {"rendered": [], "overlay": None, "colorscale": None, "error": None}
        render_note: str | None = None
        materials = _load_compare_materials(Path(args.before), before, Path(args.after), after)
        if materials is not None:
            base_a, density_a, base_b, density_b = materials
            render = _render_compare_report(
                run_dir,
                comparison=obj,
                base_a=base_a,
                density_a=density_a,
                regions_a=_region_spec_rows(before.regions),
                base_b=base_b,
                density_b=density_b,
                regions_b=_region_spec_rows(after.regions),
            )
        else:
            render_note = "源运行目录缺少 base.png/density.npy：A/B 报告渲染跳过（comparison.json 仍有效）"

        manifest_extra: dict[str, Any] = {
            "before_analysis_id": before.analysis_id,
            "after_analysis_id": after.analysis_id,
            "rendered_report_artifacts": render["rendered"],
        }
        if render["overlay"] is not None:
            # data-contract §6：共享色阶参数写入 manifest（A/B 两侧同一份记录）
            manifest_extra["overlay"] = render["overlay"]
        if render["colorscale"] is not None:
            manifest_extra["colorscale"] = render["colorscale"]
        if render["error"]:
            manifest_extra["render_error"] = render["error"]
        elif render_note:
            manifest_extra["render_note"] = render_note
        manifest = _write_manifest(
            run_dir,
            command="compare",
            computation_status="complete",
            analysis_id=None,
            extra=manifest_extra,
        )
        emit_success(
            {
                "run_directory": str(args.out),
                "comparison_schema": obj["schema_version"],
                "compatible": True,
                "exploratory": comparison.compatibility.exploratory,
                "matched_regions": len(comparison.matched),
                "added_regions": [r["id"] for r in comparison.added],
                "removed_regions": [r["id"] for r in comparison.removed],
                "artifacts": [_artifact_record(run_dir, "comparison.json").to_dict(), manifest.to_dict()],
                "limitations": list(comparison.limitations),
            }
        )
        return EXIT_OK
    except UiAttentionError:
        _cleanup_partial(run_dir)
        raise


# ---------------------------------------------------------------------------
# 参数解析与 main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ui-attention",
        description="游戏 UI 视觉注意力评估最小工具（模型结果是预测分布，不是真实眼动）",
    )
    parser.add_argument("--version", action="version", version=f"ui-attention {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="检查依赖、模型文件与哈希、设备能力及许可记录（不打印凭据）")
    p_doctor.set_defaults(func=cmd_doctor)

    p_analyze = sub.add_parser("analyze", help="截图 → 模型概率图 → AOI 统计 → 运行产物")
    p_analyze.add_argument("--request", required=True, help="AnalyzeRequest v1 JSON 文件路径")
    p_analyze.add_argument("--out", required=True, help="新运行目录（必须不存在；不覆盖历史）")
    p_analyze.add_argument(
        "--composite-background", default=None, help="透明图片的显式合成背景 'R,G,B'（缺省时带透明通道拒绝）"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_summarize = sub.add_parser("summarize", help="复用既有概率图，只按新标注重算区域统计（不重新推理）")
    p_summarize.add_argument("--analysis", required=True, help="既有运行目录（含 analysis.json 与 density.npy）")
    p_summarize.add_argument("--regions", required=True, help="独立 regions 文件（必须携带图片 SHA-256）")
    p_summarize.add_argument("--out", required=True, help="新运行目录（必须不存在）")
    p_summarize.set_defaults(func=cmd_summarize)

    p_compare = sub.add_parser("compare", help="A/B 兼容性校验 + 稳定 ID 配对 + delta_pp")
    p_compare.add_argument("--before", required=True, help="运行目录 A")
    p_compare.add_argument("--after", required=True, help="运行目录 B")
    p_compare.add_argument("--out", required=True, help="新运行目录（必须不存在）")
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_streams()  # Windows GBK 控制台兜底：中文帮助/信封不得触发 UnicodeEncodeError
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except UiAttentionError as exc:
        emit_error(exc)
        return exc.exit_code
    except BrokenPipeError:
        return EXIT_IO
    except Exception as exc:  # noqa: BLE001 - 未预期缺陷也必须结构化失败，不产生假成功
        internal = UiAttentionError(
            ErrorCode.INTERNAL_ERROR,
            f"未预期内部错误：{type(exc).__name__}: {exc}",
            {"exception_type": type(exc).__name__},
        )
        emit_error(internal)
        return EXIT_INTERNAL


# 退出码常量再导出（供测试与文档引用；语义见 data-contract §7）
__all__ = [
    "EXIT_INCOMPATIBLE",
    "EXIT_INTERNAL",
    "EXIT_INVALID_INPUT",
    "EXIT_INVALID_OUTPUT",
    "EXIT_IO",
    "EXIT_NOT_READY",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "REGISTRY_MODULE",
    "REPORT_MODULE",
    "build_parser",
    "cmd_analyze",
    "cmd_compare",
    "cmd_doctor",
    "cmd_summarize",
    "main",
    "prepare_run_directory",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
