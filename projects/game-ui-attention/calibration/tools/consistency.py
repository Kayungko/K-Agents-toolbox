"""一致性打分（calibration/protocol.md §5/§6）。

读取 package config + 回收的 responses JSON + 模型分析结果（analyze 的 run 目录，
按 ``analysis.json`` 的 ``input.image_sha256`` 对齐 density.npy），逐图/逐人指标：

- 第一眼落点密度百分位：人「第一眼落点」处密度值在全图像素密度分布中的百分位；
- 主看框 mass / area_fraction / relative_density（与技术方案 §7 同口径，框并集去重）；
- 对照基线（同口径打分）：均匀基线 U、参数化 2D 高斯中心偏置 CB（σ 按图尺寸比例）；
- 模型 vs 基线 win-rate（逐 (participant, image) 配对比较）；
- 聚合：均值/中位数，按 screen_type 分组。

输出 report.md + report.json（默认 local-data/calibration-outputs/，仓库忽略目录）。

自包含实现（仅 stdlib + numpy）：不 import ``ui_attention``；区域统计公式与
``ui_attention.metrics.regions`` 保持一致（mass=sum(P[mask])、area_fraction=area/N、
relative_density=mass/area_fraction），本模块注释明示口径来源。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "game-ui-attention-calibration-report/v1"
DENSITY_SUM_TOLERANCE = 1e-6
DEFAULT_SIGMA_RATIO = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_BASE = PROJECT_ROOT / "local-data" / "calibration-outputs"

METRICS: tuple[str, ...] = ("first_look_percentile", "box_union_mass", "box_union_relative_density")
VARIANTS: tuple[str, ...] = ("model", "cb", "uniform")

LIMITATIONS: tuple[str, ...] = (
    "粗标注≠眼动真值：真值是主观回忆式标注，存在记忆偏差（图片隐藏后作答），不得等同于眼动注视点/时长",
    "非眼动/非点击/非任务成功：第一眼落点与主看区域是主观判断，不代表真实扫视路径、点击或任务完成",
    "样本量小：不构造置信区间、不做显著性检验、不宣称统计显著；数字仅作内部方向性参考",
    "中心偏置为参数化 2D 高斯实验假设（σ 按图尺寸比例），非从人数据拟合，不得宣称为游戏玩家中心偏置",
    "观看条件简化为固定时长自由观看：不覆盖 HUD 运动、场景切换、玩家目标与熟练度",
    "绝对数值不可信：模型训练分布不含游戏 UI，本数字只回答与粗标注的相对一致性",
)

_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return float(s[n // 2]) if n % 2 else float((s[n // 2 - 1] + s[n // 2]) / 2)


def _win_rate(a: list[float], b: list[float]) -> float | None:
    """配对胜率：``#{a_i > b_i} / n``；空列表 → None。"""
    if not a:
        return None
    return sum(1 for x, y in zip(a, b, strict=True) if x > y) / len(a)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4g}"


# ---------------------------------------------------------------------------
# 密度图构造（模型侧来自 density.npy；基线侧在此构造）
# ---------------------------------------------------------------------------


def uniform_density(shape: tuple[int, int]) -> np.ndarray:
    """均匀基线 U：U(p)=1/N（benchmark-protocol §4.1）。"""
    h, w = int(shape[0]), int(shape[1])
    return np.full((h, w), 1.0 / (h * w), dtype=np.float64)


def gaussian_density(shape: tuple[int, int], center: tuple[float, float], sigma_ratio: float) -> np.ndarray:
    """参数化 2D 高斯密度，sum=1；σx=sigma_ratio·W、σy=sigma_ratio·H。"""
    h, w = int(shape[0]), int(shape[1])
    cx, cy = float(center[0]), float(center[1])
    sx = sigma_ratio * w
    sy = sigma_ratio * h
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    g = np.exp(-0.5 * (((xs[None, :] - cx) / sx) ** 2 + ((ys[:, None] - cy) / sy) ** 2))
    return g / g.sum()


def center_bias_density(shape: tuple[int, int], sigma_ratio: float) -> np.ndarray:
    """中心偏置先验 CB：2D 高斯中心 (W/2, H/2)，σ 按图尺寸比例（protocol §5）。"""
    h, w = int(shape[0]), int(shape[1])
    return gaussian_density((h, w), (w / 2.0, h / 2.0), sigma_ratio)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------


def percentile_at(density: np.ndarray, x: float, y: float) -> float:
    """模型密度在 (x, y) 最近像素处的全图像素密度百分位（(count_less + 0.5·count_eq)/N）。"""
    h, w = density.shape
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    v = density[yi, xi]
    count_less = int(np.count_nonzero(density < v))
    count_eq = int(np.count_nonzero(density == v))
    return (count_less + 0.5 * count_eq) / density.size


def box_mask(shape: tuple[int, int], boxes: list[dict[str, Any]]) -> np.ndarray:
    """多框并集去重掩码（半开 [x0, x1)，越界钳制；与技术方案 §6 掩码并集一致）。"""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0 = min(max(int(round(b["x"])), 0), w)
        y0 = min(max(int(round(b["y"])), 0), h)
        x1 = min(max(int(round(b["x"] + b["width"])), 0), w)
        y1 = min(max(int(round(b["y"] + b["height"])), 0), h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def box_stats(density: np.ndarray, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """框并集统计：mass / area_fraction / relative_density（口径与 metrics.regions 一致）。"""
    mask = box_mask(density.shape, boxes)
    area_px = int(mask.sum())
    total = int(density.size)
    if area_px == 0:
        return {"mass": 0.0, "area_fraction": 0.0, "relative_density": 0.0, "area_px": 0}
    mass = float(density[mask].sum())
    area_fraction = area_px / total
    return {
        "mass": mass,
        "area_fraction": area_fraction,
        "relative_density": mass / area_fraction,
        "area_px": area_px,
    }


def compute_pair(
    point: dict[str, Any],
    boxes: list[dict[str, Any]],
    model_density: np.ndarray,
    cb_density: np.ndarray,
    uniform: np.ndarray,
) -> dict[str, Any]:
    """单 (participant, image) 配对的模型/基线同口径打分。"""
    x = float(point["x"])
    y = float(point["y"])
    m_box = box_stats(model_density, boxes)
    c_box = box_stats(cb_density, boxes)
    u_box = box_stats(uniform, boxes)
    return {
        "first_look_percentile_model": percentile_at(model_density, x, y),
        "first_look_percentile_cb": percentile_at(cb_density, x, y),
        "first_look_percentile_uniform": percentile_at(uniform, x, y),
        "box_union_mass_model": m_box["mass"],
        "box_union_mass_cb": c_box["mass"],
        "box_union_mass_uniform": u_box["mass"],
        "box_union_relative_density_model": m_box["relative_density"],
        "box_union_relative_density_cb": c_box["relative_density"],
        "box_union_relative_density_uniform": u_box["relative_density"],
        "box_union_area_fraction": m_box["area_fraction"],
        "box_union_area_px": m_box["area_px"],
    }


# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------


def load_package_config(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("images"), list):
        raise ValueError("config.json 缺少 images 清单")
    return obj


def load_responses(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() == ".json")
    elif p.is_file():
        files = [p]
    else:
        raise ValueError(f"responses 路径不存在：{path}")
    out: list[dict[str, Any]] = []
    for f in files:
        obj = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("images"), list):
            out.append(obj)
    return out


def validate_density(arr: Any, label: str) -> np.ndarray:
    d = np.asarray(arr, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        raise ValueError(f"density 必须是非空 2D 数组（{label}），shape={getattr(d, 'shape', None)}")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"density 含 NaN/Inf（{label}）")
    if np.any(d < 0):
        raise ValueError(f"density 含负值（{label}）")
    if abs(float(d.sum()) - 1.0) > DENSITY_SUM_TOLERANCE:
        raise ValueError(f"density sum={float(d.sum())} 与 1 偏差超容差（{label}）")
    return d


def load_model_densities(runs_dir: str | Path) -> dict[str, dict[str, Any]]:
    """扫描 runs 目录下的 analyze run（analysis.json + density.npy），按 image_sha256 建表。"""
    root = Path(runs_dir)
    if not root.is_dir():
        raise ValueError(f"runs 目录不存在：{runs_dir}")
    out: dict[str, dict[str, Any]] = {}
    for analysis_path in sorted(root.rglob("analysis.json")):
        run_dir = analysis_path.parent
        density_path = run_dir / "density.npy"
        if not density_path.is_file():
            continue
        obj = json.loads(analysis_path.read_text(encoding="utf-8"))
        inp = obj.get("input") or {}
        sha = inp.get("image_sha256")
        w = inp.get("width")
        h = inp.get("height")
        if not sha or not w or not h:
            continue
        density = validate_density(np.load(density_path), str(sha))
        if density.shape != (int(h), int(w)):
            raise ValueError(f"run {run_dir.name}: density 形状 {density.shape} != analysis 记录 {(int(h), int(w))}")
        if sha in out:
            raise ValueError(f"同一 sha256 出现多个 run 目录（冲突）：{sha}")
        out[str(sha)] = {"density": density, "width": int(w), "height": int(h)}
    return out


# ---------------------------------------------------------------------------
# 排除标准与聚合
# ---------------------------------------------------------------------------


def _point_in_bounds(point: dict[str, Any], w: int, h: int) -> bool:
    x = point.get("x")
    y = point.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    return 0.0 <= float(x) < w and 0.0 <= float(y) < h


def _looks_random(entries: list[dict[str, Any]]) -> bool:
    """乱点自检：所有图第一眼落点完全相同且主看框完全相同（protocol §7.3）。"""
    answered = [e for e in entries if e.get("responded") and e.get("first_look") and e.get("boxes")]
    if len(answered) < 2:
        return False

    def _pt(e: dict[str, Any]) -> tuple[float, float]:
        return (round(float(e["first_look"]["x"]), 1), round(float(e["first_look"]["y"]), 1))

    def _boxes(e: dict[str, Any]) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(
            sorted(
                (
                    round(float(b["x"]), 1),
                    round(float(b["y"]), 1),
                    round(float(b["width"]), 1),
                    round(float(b["height"]), 1),
                )
                for b in e["boxes"]
            )
        )

    first_pt, first_bx = _pt(answered[0]), _boxes(answered[0])
    return all(_pt(e) == first_pt for e in answered) and all(_boxes(e) == first_bx for e in answered)


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(pairs)}
    if pairs:
        out["box_union_area_fraction_mean"] = _mean([p["box_union_area_fraction"] for p in pairs])
    for metric in METRICS:
        m: dict[str, Any] = {}
        for variant in VARIANTS:
            vals = [p[f"{metric}_{variant}"] for p in pairs]
            m[f"{variant}_mean"] = _mean(vals)
            m[f"{variant}_median"] = _median(vals)
        m["model_vs_cb_win_rate"] = _win_rate(
            [p[f"{metric}_model"] for p in pairs], [p[f"{metric}_cb"] for p in pairs]
        )
        m["model_vs_uniform_win_rate"] = _win_rate(
            [p[f"{metric}_model"] for p in pairs], [p[f"{metric}_uniform"] for p in pairs]
        )
        out[metric] = m
    return out


def evaluate(
    package_config: dict[str, Any],
    responses: list[dict[str, Any]],
    model_densities: dict[str, dict[str, Any]],
    sigma_ratio: float,
) -> dict[str, Any]:
    """主计算：排除 → 逐配对打分 → 聚合；返回报告 dict（含必写 limitations）。"""
    images_cfg = package_config.get("images", [])
    config_by_sha = {im["sha256"]: im for im in images_cfg}
    total_images = len(images_cfg)

    pairs: list[dict[str, Any]] = []
    excluded_participants: list[dict[str, Any]] = []
    excluded_images: list[dict[str, Any]] = []
    missing_model: list[dict[str, Any]] = []

    for resp in responses:
        pid = resp.get("participant_id") or "?"
        entries = resp.get("images") or []
        if not resp.get("completed", False):
            excluded_participants.append({"participant_id": pid, "reason": "completed=false"})
            continue
        if len(entries) != total_images:
            excluded_participants.append({"participant_id": pid, "reason": f"图片数 {len(entries)} != {total_images}"})
            continue
        if _looks_random(entries):
            excluded_participants.append({"participant_id": pid, "reason": "乱点自检：所有图落点与框完全相同"})
            continue

        for entry in entries:
            sha = entry.get("sha256")
            cfg = config_by_sha.get(sha)
            if cfg is None:
                excluded_images.append({"participant_id": pid, "sha256": sha, "reason": "sha256 不在 config 中"})
                continue
            if not entry.get("responded", False):
                excluded_images.append({"participant_id": pid, "sha256": sha, "reason": "未作答"})
                continue
            point = entry.get("first_look")
            boxes = entry.get("boxes") or []
            if not point or not boxes:
                excluded_images.append({"participant_id": pid, "sha256": sha, "reason": "缺第一眼落点或主看框"})
                continue
            w, h = int(cfg["width"]), int(cfg["height"])
            if not _point_in_bounds(point, w, h):
                excluded_images.append({"participant_id": pid, "sha256": sha, "reason": "落点越界"})
                continue
            model = model_densities.get(sha)
            if model is None:
                missing_model.append({"participant_id": pid, "sha256": sha})
                continue
            m = compute_pair(
                point, boxes, model["density"], center_bias_density((h, w), sigma_ratio), uniform_density((h, w))
            )
            m["participant_id"] = pid
            m["sha256"] = sha
            m["screen_type"] = cfg.get("screen_type", "unknown")
            m["filename"] = cfg.get("filename") or cfg.get("source_name")
            pairs.append(m)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": _utcnow(),
        "study_id": package_config.get("study_id"),
        "sigma_ratio": sigma_ratio,
        "n_images_in_package": total_images,
        "n_responses": len(responses),
        "n_participants_included": len({p["participant_id"] for p in pairs}),
        "n_pairs": len(pairs),
        "excluded_participants": excluded_participants,
        "excluded_images": excluded_images,
        "missing_model_images": missing_model,
        "aggregate": _aggregate(pairs),
        "by_screen_type": {},
        "limitations": list(LIMITATIONS),
    }
    for st in sorted({p["screen_type"] for p in pairs}):
        report["by_screen_type"][st] = _aggregate([p for p in pairs if p["screen_type"] == st])
    return report


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 粗标注一致性校准报告")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{report['created_at_utc']}")
    lines.append(f"- study_id：{report.get('study_id')}")
    lines.append(f"- 中心偏置 σ 比例：{report['sigma_ratio']}（σx=σ·W、σy=σ·H）")
    lines.append(
        f"- 包内图片 {report['n_images_in_package']}；回收 responses {report['n_responses']}；"
        f"纳入参与者 {report['n_participants_included']}；纳入配对 {report['n_pairs']}"
    )
    lines.append("")
    lines.append("> 本报告为**内部粗标注一致性**证据，非眼动验证；数字不构成预测准确率/命中率声明。")
    lines.append("")

    lines.append("## 汇总（模型 vs 基线，同口径）")
    lines.append("")
    lines.append("| 指标 | model 均值 | cb 均值 | uniform 均值 | model 中位 | win-rate vs cb | win-rate vs uniform |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    agg = report["aggregate"]
    for metric in METRICS:
        m = agg[metric]
        lines.append(
            f"| {metric} | {_fmt(m['model_mean'])} | {_fmt(m['cb_mean'])} | {_fmt(m['uniform_mean'])} | "
            f"{_fmt(m['model_median'])} | {_fmt(m['model_vs_cb_win_rate'])} | {_fmt(m['model_vs_uniform_win_rate'])} |"
        )
    lines.append("")
    lines.append(f"- 主看框面积占比（并集）均值：{_fmt(agg.get('box_union_area_fraction_mean'))}")
    lines.append("- uniform 基线：任意点百分位恒为 0.5；框 mass=area_fraction、relative_density=1。")
    lines.append("")

    by_st = report.get("by_screen_type") or {}
    if by_st:
        lines.append("## 按界面类型分组")
        lines.append("")
        for st in sorted(by_st):
            lines.append(f"### {st}")
            lines.append("")
            lines.append("| 指标 | model 均值 | cb 均值 | uniform 均值 | win-rate vs cb | win-rate vs uniform |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for metric in METRICS:
                m = by_st[st][metric]
                lines.append(
                    f"| {metric} | {_fmt(m['model_mean'])} | {_fmt(m['cb_mean'])} | {_fmt(m['uniform_mean'])} | "
                    f"{_fmt(m['model_vs_cb_win_rate'])} | {_fmt(m['model_vs_uniform_win_rate'])} |"
                )
            lines.append("")

    lines.append("## 排除与缺失")
    lines.append("")
    lines.append(f"- 排除参与者：{len(report['excluded_participants'])} 人")
    for e in report["excluded_participants"]:
        lines.append(f"  - {e['participant_id']}: {e['reason']}")
    lines.append(f"- 排除图片（未作答/缺标注/越界/错图）：{len(report['excluded_images'])} 张")
    for e in report["excluded_images"]:
        lines.append(f"  - {e['participant_id']}/{e.get('sha256', '?')}: {e['reason']}")
    lines.append(f"- 无模型结果的图片（未跑 analyze）：{len(report['missing_model_images'])} 张")
    lines.append("")

    lines.append("## 局限性（必写）")
    lines.append("")
    for i, lim in enumerate(report["limitations"], 1):
        lines.append(f"{i}. {lim}")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return md_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_out_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUT_BASE / f"calibration-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="consistency.py", description="内部粗标注一致性打分（模型 vs 基线）")
    parser.add_argument("--package-config", required=True, help="研究包 config.json 路径")
    parser.add_argument("--responses", required=True, help="回收的 responses JSON 目录（或单文件）")
    parser.add_argument("--runs", required=True, help="analyze 运行目录（含 density.npy + analysis.json）")
    parser.add_argument("--sigma-ratio", type=float, default=DEFAULT_SIGMA_RATIO, help="中心偏置 σ 比例（默认 0.25）")
    parser.add_argument(
        "--out", type=Path, default=None, help="输出目录（默认 local-data/calibration-outputs/<时间戳>）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package_config = load_package_config(args.package_config)
        responses = load_responses(args.responses)
        model_densities = load_model_densities(args.runs)
        report = evaluate(package_config, responses, model_densities, args.sigma_ratio)
        out = Path(args.out) if args.out is not None else default_out_dir()
        md_path, json_path = write_report(report, out)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out),
                "report_md": str(md_path),
                "report_json": str(json_path),
                "n_pairs": report["n_pairs"],
                "n_participants_included": report["n_participants_included"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
