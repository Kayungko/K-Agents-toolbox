"""analysis v1 schema（data-contract.md §4 计算结果与证据）。

字段与 §4 表逐一对应；另含 C1 扩展的可选字段（created_at_utc、metrics_version、
player_goal、region_union），扩展字段用于 summarize/compare 的可复现校验，
不改变 §4 必备字段语义。计算完成不代表评审完成，也不代表模型预测准确。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ErrorCode, UiAttentionError

ANALYSIS_SCHEMA_VERSION = "game-ui-attention-analysis/v1"
METRICS_VERSION = "aoi-metrics/v1"  # AOI 统计口径版本（compare 兼容性检查消费）

COMPUTATION_STATUS: tuple[str, ...] = ("complete", "failed")
REVIEW_STATUS: tuple[str, ...] = ("not_requested", "pending", "complete", "failed")
EVIDENCE_TYPE = "model_prediction"

_ANALYSIS_KEYS = {
    "schema_version",
    "analysis_id",
    "computation_status",
    "review_status",
    "evidence_type",
    "created_at_utc",
    "metrics_version",
    "player_goal",
    "input",
    "model",
    "profile",
    "runtime",
    "regions",
    "region_union",
    "limitations",
    "artifacts",
    "errors",
}
_INPUT_KEYS = {"image_path", "image_sha256", "width", "height", "inference_width", "inference_height", "screen_type"}
# model 键：§4 要求（backend_id/version/weights_sha256/code_version/capabilities）
# + C1 证据扩展（native_semantics/license_status，来自 BackendInfo 冻结字段，供审计与报告消费）
_MODEL_KEYS = {
    "backend_id",
    "version",
    "weights_sha256",
    "code_version",
    "capabilities",
    "native_semantics",
    "license_status",
}
_PROFILE_KEYS = {"profile_name", "config_hash", "preprocessing", "centerbias", "viewing_conditions"}
_RUNTIME_KEYS = {"dependencies", "device", "precision", "elapsed_ms", "peak_mem_mb"}
_REGION_RESULT_KEYS = {
    "id",
    "label",
    "role",
    "source",
    "status",
    "geometry",
    "area_px",
    "area_fraction",
    "probability_mass",
    "relative_density",
}
_ARTIFACT_KEYS = {"path", "sha256", "size_bytes"}


def _require(cond: bool, errors: list[str], message: str) -> None:
    if not cond:
        errors.append(message)


@dataclass(frozen=True)
class ArtifactRecord:
    """产物：相对路径及哈希；缺少的产物不列为成功（§4 artifacts 行）。"""

    path: str
    sha256: str
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.size_bytes is not None:
            obj["size_bytes"] = self.size_bytes
        return obj

    @classmethod
    def from_dict(cls, obj: Any, errors: list[str], where: str) -> ArtifactRecord:
        if not isinstance(obj, dict):
            errors.append(f"{where} 必须是对象")
            return cls(path="", sha256="")
        unknown = set(obj) - _ARTIFACT_KEYS
        _require(not unknown, errors, f"{where} 含未知字段 {sorted(unknown)}")
        _require(isinstance(obj.get("path"), str) and obj.get("path"), errors, f"{where}.path 必须是非空相对路径")
        _require(
            isinstance(obj.get("sha256"), str) and len(obj["sha256"]) == 64, errors, f"{where}.sha256 必须是 64 位哈希"
        )
        return cls(path=obj.get("path", ""), sha256=obj.get("sha256", ""), size_bytes=obj.get("size_bytes"))


@dataclass(frozen=True)
class AnalysisRecord:
    """analysis.json 的内存形态（§4 全字段）。"""

    analysis_id: str
    computation_status: str
    input: dict[str, Any]
    model: dict[str, Any]
    profile: dict[str, Any]
    runtime: dict[str, Any]
    regions: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    review_status: str = "not_requested"
    evidence_type: str = EVIDENCE_TYPE
    schema_version: str = ANALYSIS_SCHEMA_VERSION
    created_at_utc: str | None = None
    metrics_version: str = METRICS_VERSION
    player_goal: str | None = None
    region_union: dict[str, Any] | None = field(default=None)

    # -- 校验 ---------------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []
        _require(
            self.schema_version == ANALYSIS_SCHEMA_VERSION,
            errors,
            f"schema_version 必须是 {ANALYSIS_SCHEMA_VERSION!r}，得到 {self.schema_version!r}",
        )
        _require(isinstance(self.analysis_id, str) and bool(self.analysis_id), errors, "analysis_id 必须是非空字符串")
        _require(
            self.computation_status in COMPUTATION_STATUS,
            errors,
            f"computation_status 必须是 {COMPUTATION_STATUS}，得到 {self.computation_status!r}",
        )
        _require(
            self.review_status in REVIEW_STATUS,
            errors,
            f"review_status 必须是 {REVIEW_STATUS}，得到 {self.review_status!r}",
        )
        _require(
            self.evidence_type == EVIDENCE_TYPE,
            errors,
            f"evidence_type 必须是 {EVIDENCE_TYPE!r}（模型结果是预测分布，不是真实眼动）",
        )
        self._validate_input(errors)
        self._validate_model(errors)
        self._validate_profile(errors)
        self._validate_runtime(errors)
        for i, r in enumerate(self.regions):
            self._validate_region(r, i, errors)
        for i, a in enumerate(self.artifacts):
            artifact_errors: list[str] = []
            ArtifactRecord.from_dict(a.to_dict(), artifact_errors, f"artifacts[{i}]")
            errors.extend(artifact_errors)
        if errors:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "analysis 记录校验失败", {"errors": errors})

    def _validate_input(self, errors: list[str]) -> None:
        inp = self.input
        if not isinstance(inp, dict):
            errors.append("input 必须是 dict")
            return
        unknown = set(inp) - _INPUT_KEYS
        _require(not unknown, errors, f"input 含未知字段 {sorted(unknown)}")
        sha = inp.get("image_sha256")
        _require(isinstance(sha, str) and len(sha) == 64, errors, "input.image_sha256 必须是 64 位图片哈希")
        for key in ("width", "height", "inference_width", "inference_height"):
            v = inp.get(key)
            _require(isinstance(v, int) and not isinstance(v, bool) and v > 0, errors, f"input.{key} 必须是正整数")

    def _validate_model(self, errors: list[str]) -> None:
        m = self.model
        if not isinstance(m, dict):
            errors.append("model 必须是 dict")
            return
        unknown = set(m) - _MODEL_KEYS
        _require(not unknown, errors, f"model 含未知字段 {sorted(unknown)}")
        _require(
            isinstance(m.get("backend_id"), str) and bool(m.get("backend_id")),
            errors,
            "model.backend_id 必须是非空字符串",
        )
        _require(isinstance(m.get("version"), str) and bool(m.get("version")), errors, "model.version 必须是非空字符串")
        hashes = m.get("weights_sha256")
        _require(
            isinstance(hashes, list) and all(isinstance(h, str) and len(h) == 64 for h in hashes),
            errors,
            "model.weights_sha256 必须是全部权重哈希 list[str]",
        )
        _require(isinstance(m.get("capabilities"), list), errors, "model.capabilities 必须是 list[str]")

    def _validate_profile(self, errors: list[str]) -> None:
        p = self.profile
        if not isinstance(p, dict):
            errors.append("profile 必须是 dict")
            return
        unknown = set(p) - _PROFILE_KEYS
        _require(not unknown, errors, f"profile 含未知字段 {sorted(unknown)}")
        _require(
            isinstance(p.get("profile_name"), str) and bool(p.get("profile_name")),
            errors,
            "profile.profile_name 必须是实际执行的登记名（不能只保存请求别名）",
        )
        _require(
            isinstance(p.get("config_hash"), str) and len(p.get("config_hash", "")) == 64,
            errors,
            "profile.config_hash 必须是 64 位合成哈希",
        )
        _require(isinstance(p.get("preprocessing"), dict), errors, "profile.preprocessing 必须是 dict")
        _require(
            p.get("centerbias") is None or isinstance(p.get("centerbias"), dict),
            errors,
            "profile.centerbias 必须是 dict|None",
        )
        _require(isinstance(p.get("viewing_conditions"), dict), errors, "profile.viewing_conditions 必须是 dict")

    def _validate_runtime(self, errors: list[str]) -> None:
        rt = self.runtime
        if not isinstance(rt, dict):
            errors.append("runtime 必须是 dict")
            return
        unknown = set(rt) - _RUNTIME_KEYS
        _require(not unknown, errors, f"runtime 含未知字段 {sorted(unknown)}")
        _require(isinstance(rt.get("dependencies"), dict), errors, "runtime.dependencies 必须是依赖版本 dict[str,str]")
        _require(rt.get("device") in ("cpu", "cuda"), errors, "runtime.device 必须是 cpu|cuda（实测）")

    def _validate_region(self, r: Any, i: int, errors: list[str]) -> None:
        where = f"regions[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{where} 必须是对象")
            return
        unknown = set(r) - _REGION_RESULT_KEYS
        _require(not unknown, errors, f"{where} 含未知字段 {sorted(unknown)}")
        _require(isinstance(r.get("id"), str) and bool(r.get("id")), errors, f"{where}.id 必须是非空字符串")
        _require(isinstance(r.get("geometry"), dict), errors, f"{where}.geometry 必须是边界对象")
        _require(
            isinstance(r.get("area_px"), int) and not isinstance(r.get("area_px"), bool) and r.get("area_px", 0) > 0,
            errors,
            f"{where}.area_px 必须是正整数（像素掩码计数）",
        )
        for key in ("probability_mass", "area_fraction", "relative_density"):
            v = r.get(key)
            _require(isinstance(v, (int, float)) and not isinstance(v, bool), errors, f"{where}.{key} 必须是浮点原始值")
        mass = r.get("probability_mass")
        if isinstance(mass, (int, float)) and not (0.0 <= float(mass) <= 1.0 + 1e-9):
            errors.append(f"{where}.probability_mass 超出 [0,1]：{mass}")

    # -- 序列化 ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "computation_status": self.computation_status,
            "review_status": self.review_status,
            "evidence_type": self.evidence_type,
            "metrics_version": self.metrics_version,
            "input": dict(self.input),
            "model": dict(self.model),
            "profile": dict(self.profile),
            "runtime": dict(self.runtime),
            "regions": [dict(r) for r in self.regions],
            "limitations": list(self.limitations),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "errors": [dict(e) for e in self.errors],
        }
        if self.created_at_utc is not None:
            obj["created_at_utc"] = self.created_at_utc
        if self.player_goal is not None:
            obj["player_goal"] = self.player_goal
        if self.region_union is not None:
            obj["region_union"] = self.region_union
        return obj

    @classmethod
    def from_dict(cls, obj: Any) -> AnalysisRecord:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "analysis 必须是 JSON 对象")
        version = obj.get("schema_version")
        if version != ANALYSIS_SCHEMA_VERSION:
            raise UiAttentionError(
                ErrorCode.INVALID_SCHEMA_VERSION,
                f"未知 schema_version {version!r}（本实现只接受 {ANALYSIS_SCHEMA_VERSION!r}）",
                {"got": version, "expected": ANALYSIS_SCHEMA_VERSION},
            )
        unknown = set(obj) - _ANALYSIS_KEYS
        if unknown:
            raise UiAttentionError(ErrorCode.INVALID_REQUEST, "analysis 含未知字段", {"unknown": sorted(unknown)})
        record = cls(
            analysis_id=obj.get("analysis_id"),  # type: ignore[arg-type]
            computation_status=obj.get("computation_status"),  # type: ignore[arg-type]
            input=dict(obj.get("input", {})),
            model=dict(obj.get("model", {})),
            profile=dict(obj.get("profile", {})),
            runtime=dict(obj.get("runtime", {})),
            regions=tuple(dict(r) for r in obj.get("regions", ())),
            limitations=tuple(obj.get("limitations", ())),
            artifacts=tuple(
                ArtifactRecord(path=a.get("path", ""), sha256=a.get("sha256", ""), size_bytes=a.get("size_bytes"))
                for a in obj.get("artifacts", ())
                if isinstance(a, dict)
            ),
            errors=tuple(dict(e) for e in obj.get("errors", ())),
            review_status=obj.get("review_status", "not_requested"),
            evidence_type=obj.get("evidence_type", EVIDENCE_TYPE),
            schema_version=version,
            created_at_utc=obj.get("created_at_utc"),
            metrics_version=obj.get("metrics_version", METRICS_VERSION),
            player_goal=obj.get("player_goal"),
            region_union=obj.get("region_union"),
        )
        record.validate()
        return record


def load_analysis(run_dir: str | Path) -> AnalysisRecord:
    """读取 <run-directory>/analysis.json 并校验。"""
    p = Path(run_dir) / "analysis.json"
    if not p.is_file():
        raise UiAttentionError(
            ErrorCode.ARTIFACT_CHECK_FAILED,
            f"analysis.json 不存在于运行目录：{Path(run_dir).name}",
        )
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise UiAttentionError(ErrorCode.ARTIFACT_CHECK_FAILED, "analysis.json 读取失败", {"reason": str(exc)}) from exc
    return AnalysisRecord.from_dict(obj)
