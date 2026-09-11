"""Backend profile registry (C2 scope).

Implements the frozen profile-registration rules of ``docs/data-contract.md``
§3 (2026-09-11 freeze) on top of the C1-owned contract types
(:class:`ui_attention.contracts.backend.ResolvedProfile` / ``WeightRef``):

- ``backend_profile`` 必须是本模块已登记、不可在执行时静默改写的配置；
  登记名规范 ``<backend>-<variant>-v<N>``，首个真实登记为
  ``foveacast-onnx-3s-v1``（G0）；G2 增补窗口敏感性变体
  ``foveacast-onnx-1s-v1`` / ``foveacast-onnx-7s-v1``（许可字段同 3s）。
- 登记项包含：backend_id+version、权重清单与哈希、预处理参数（目标尺寸、
  值域、均值处理）、centerbias 引用（foveacast 无 → ``None``）、观看条件
  假设、许可状态与全部配置的合成哈希（``config_hash``，64 位小写十六进制，
  满足 C1 ``ResolvedProfile.validate()`` 的 sha256 校验）。
- 模块级接口对齐 C1 ``BackendRegistry`` Protocol（cli.py 动态导入面）：
  :func:`list_profiles` / :func:`resolve_profile` / :func:`get_backend` /
  :func:`doctor_report`。返回的 ``ResolvedProfile`` 为 frozen dataclass
  （只读）；运行时改写会被 :func:`verify_profile_integrity` 的合成哈希复核
  拒绝（``PROFILE_NOT_REGISTERED``，退出码 3，对齐 Protocol 文档口径）。
- 分析记录必须保存实际执行的解析结果（``ResolvedProfile.to_dict()``），
  不能只保存请求中的别名。
- 许可状态与 CC-BY-4.0 传递署名作为登记元数据保存在
  :class:`ProfileRegistration`（:data:`LICENSE_RECORD` / :data:`ATTRIBUTION`），
  供 doctor 与报告渲染消费；BackendInfo.license_status 使用冻结 4 键字典。

许可与署名（G0 结论，R1 model-candidates.md §C2/§6）：代码 MIT 已验证；
权重 = MIT + UEyes CC-BY-4.0 署名传递（下游必须携带 Jiang et al. 2023
引用）；商用链残留缺口 G3（SALICON 条款未核实）/G4（原始 VGG16 权重条款
未核实）/G8（ImageNet 类预训练残留风险）在打包/商用前必须关闭；当前
``cleared_for="internal-eval"``。完整引用见 :data:`ATTRIBUTION`。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ui_attention.contracts.backend import Backend, ResolvedProfile, WeightRef

from .errors import ProfileRejectedError

__all__ = [
    "ATTRIBUTION",
    "CENTERBIAS_NOTE",
    "DEEPGAZE_ATTRIBUTION",
    "DEEPGAZE_CENTERBIAS_WEIGHT",
    "DEEPGAZE_IIE_BACKEND_ID",
    "DEEPGAZE_IIE_CENTERBIAS",
    "DEEPGAZE_IIE_LICENSE_STATUS",
    "DEEPGAZE_IIE_PREPROCESSING",
    "DEEPGAZE_IIE_PROFILE",
    "DEEPGAZE_IIE_VIEWING_CONDITIONS",
    "DEEPGAZE_IIE_WEIGHT",
    "DEEPGAZE_LICENSE_RECORD",
    "DEEPGAZE_SOURCE_PIN",
    "FOVEACAST_1S_BACKEND_ID",
    "FOVEACAST_1S_PROFILE",
    "FOVEACAST_1S_VIEWING_CONDITIONS",
    "FOVEACAST_1S_WEIGHT",
    "FOVEACAST_3S_BACKEND_ID",
    "FOVEACAST_3S_BACKEND_VERSION",
    "FOVEACAST_3S_PREPROCESSING",
    "FOVEACAST_3S_PROFILE",
    "FOVEACAST_3S_VIEWING_CONDITIONS",
    "FOVEACAST_3S_WEIGHT",
    "FOVEACAST_7S_BACKEND_ID",
    "FOVEACAST_7S_PROFILE",
    "FOVEACAST_7S_VIEWING_CONDITIONS",
    "FOVEACAST_7S_WEIGHT",
    "FOVEACAST_BACKEND_IDS",
    "FOVEACAST_LICENSE_STATUS",
    "LICENSE_RECORD",
    "ProfileRegistration",
    "compute_config_hash",
    "doctor_report",
    "get_backend",
    "license_record",
    "list_profiles",
    "profile_for_backend",
    "registration_for",
    "resolve_profile",
    "verify_profile_integrity",
]

FOVEACAST_3S_BACKEND_ID = "foveacast-onnx-3s"
FOVEACAST_1S_BACKEND_ID = "foveacast-onnx-1s"
FOVEACAST_7S_BACKEND_ID = "foveacast-onnx-7s"
FOVEACAST_BACKEND_IDS: tuple[str, ...] = (
    FOVEACAST_1S_BACKEND_ID,
    FOVEACAST_3S_BACKEND_ID,
    FOVEACAST_7S_BACKEND_ID,
)
FOVEACAST_3S_BACKEND_VERSION = "0.1.0"
FOVEACAST_3S_PROFILE = "foveacast-onnx-3s-v1"
FOVEACAST_1S_PROFILE = "foveacast-onnx-1s-v1"
FOVEACAST_7S_PROFILE = "foveacast-onnx-7s-v1"

#: DeepGaze IIE 内部评估对照线（G1' 批准 2026-09-11：仅限内部研究评估，
#: 不打包不分发；许可缺口 G1/G2/G5 未闭合）。
DEEPGAZE_IIE_BACKEND_ID = "deepgaze-iie"
DEEPGAZE_IIE_BACKEND_VERSION = "0.1.0"
DEEPGAZE_IIE_PROFILE = "deepgaze-iie-v1"
DEEPGAZE_SOURCE_PIN = "c7db17e2d1d7ea6468ffdee2cfaddf141095dcff"

_RELEASE_BASE = "https://github.com/khawkins98/foveacast-training/releases/download/v0.2.0/"

#: Registered weight manifest (3s = 主用窗口, G0 首登记). Expected sha256
#: transcribed from the GitHub release page (khawkins98/foveacast-training
#: v0.2.0, expanded_assets digest, fetched 2026-09-11); cross-checked with
#: model-candidates.md §C2 prefix ``842a23f97908d146`` and runtime-feasibility
#: §4 #2 suffix ``585ef76e``. size_bytes = exact locally verified file size
#: after the hash-gated download (GitHub UI only publishes the rounded
#: "53.9 MB"); the sha256 remains the authoritative gate either way.
FOVEACAST_3S_WEIGHT = WeightRef(
    name="foveacast-v3-3s-fp16.onnx",
    source_url=_RELEASE_BASE + "foveacast-v3-3s-fp16.onnx",
    sha256="842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e",
    size_bytes=56549242,
)

#: 1s/7s 窗口敏感性权重（G2 批准下载，2026-09-11）。sha256 取自同一
#: expanded_assets digest 抓取（本会话二次复核，与 R1 §C2 前缀互证）；
#: size_bytes 为下载后 Get-FileHash/Length 实测值（同架构同尺寸）。
FOVEACAST_1S_WEIGHT = WeightRef(
    name="foveacast-v3-1s-fp16.onnx",
    source_url=_RELEASE_BASE + "foveacast-v3-1s-fp16.onnx",
    sha256="4b9fdc2734e36c612a120ab7b0050ae276723160ccc625c6f554e906dd6345d5",
    size_bytes=56549242,
)
FOVEACAST_7S_WEIGHT = WeightRef(
    name="foveacast-v3-7s-fp16.onnx",
    source_url=_RELEASE_BASE + "foveacast-v3-7s-fp16.onnx",
    sha256="cf66388dc6fe5db4712c77cf3929d04380ed73ac963677b8c9de2b6380fb51e0",
    size_bytes=56549242,
)

#: License status exactly in the frozen BackendInfo.license_status shape
#: (data-contract §3: {"code", "weights", "gaps", "cleared_for"}).
FOVEACAST_LICENSE_STATUS: dict[str, Any] = {
    "code": "MIT (已验证: foveacast-training LICENSE, (c) 2026 Ken Hawkins; "
    "上游 MSI-Net 移植 alexanderkroner/saliency MIT, (c) 2019 Alexander Kroner)",
    "weights": "MIT + UEyes CC-BY-4.0 署名传递 (作者声明: 发布产物为 MIT 代码, "
    "下游使用必须携带 Jiang et al. 2023 引用)",
    "gaps": ["G3", "G4", "G8"],
    "cleared_for": "internal-eval",
}

#: Full license record for doctor output / report rendering (richer than the
#: frozen 4-key contract dict; evidence trail from R1, 核查日期 2026-09-11).
LICENSE_RECORD: dict[str, Any] = {
    "status": FOVEACAST_LICENSE_STATUS,
    "evidence": {
        "code": "raw LICENSE 全文已验证 (R1 model-candidates.md §C2 许可四件套)",
        "weights": "README 'Licences, in one place' 原文: Released model artefacts — "
        "MIT code, but downstream use must carry the UEyes citation per CC BY 4.0",
        "training_data": "UEyes dataset, Zenodo record 8010312, CC-BY-4.0 "
        "(Zenodo API license.id 已验证), 允许商用、须署名",
        "backbone": "MSI-Net (alexanderkroner/saliency) 代码 MIT 已验证; "
        "HF 权重卡 license: mit 已验证",
    },
    "gaps_detail": {
        "G3": "SALICON 训练数据条款未经一手核实 (官网不可达); foveacast 权重自 "
        "SALICON 预训练 stock 权重微调而来 → 商用链残留",
        "G4": "原始 VGG16 预训练权重条款 (通称 Oxford VGG CC-BY-4.0) 未从一手来源核实",
        "G8": "ImageNet 预训练骨干通用残留风险, 商用发布时统一记录披露口径",
    },
    "obligation": "内部评估已放行 (cleared_for=internal-eval); 打包/分发/商用前必须"
    "先关闭 G3/G4/G8 (一级总控裁决项)。任何下游产物必须携带 ATTRIBUTION 两条引用"
    " (CC-BY-4.0 传递义务)。",
    "attribution": None,  # filled below (module-level constant ordering)
    "cleared_for": "internal-eval",
}

#: CC-BY-4.0 传递署名义务 —— 完整引用 (来源: foveacast-training README
#: Attribution 节 / Zenodo 8010312 / DOI, R1 已验证)。onnx_foveacast.py 模块
#: docstring 同步记录。
ATTRIBUTION: tuple[str, ...] = (
    "Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B., "
    "Kylmälä, J., & Oulasvirta, A. (2023). UEyes: Understanding Visual Saliency "
    "across User Interface Types. In Proceedings of the 2023 CHI Conference on "
    "Human Factors in Computing Systems (CHI '23), Article 285, 1-21. "
    "DOI: 10.1145/3544548.3581096. Dataset: Zenodo record 8010312, CC-BY-4.0.",
    "Kroner, A., Senden, M., Driessens, K., & Goebel, R. (2020). Contextual "
    "Encoder-Decoder Network for Visual Saliency Prediction. Neural Networks, "
    "129, 261-270. DOI: 10.1016/j.neunet.2020.05.004. "
    "(MSI-Net architecture, MIT)",
)
LICENSE_RECORD["attribution"] = list(ATTRIBUTION)

#: 预处理参数 (technical-design §3 冻结口径: 输入 (N,3,240,320) float32 RGB
#: [0,255], 均值减除在图内完成, 调用方不得预处理; 输出逐图 min-max [0,1],
#: 适配层 sum=1 重归一化后声明 probability_density)。
FOVEACAST_3S_PREPROCESSING: dict[str, Any] = {
    "target_height": 240,
    "target_width": 320,
    "input_layout": "NCHW",
    "channel_order": "RGB",
    "input_dtype": "float32",
    "value_range": [0.0, 255.0],
    "mean_subtraction": "in_graph",  # 均值减除在 ONNX 图内, 适配层不做
    "mean_subtraction_note": "Kroner 特例均值 (103.939, 116.779, 123.68) 按 RGB 序"
    "在图内减除 (R1 已验证 msinet.py); 调用方传入原始 [0,255] 值",
    "resize_filter": "pillow_bicubic",  # 对齐上游 README quickstart 默认 resize
    "resize_policy": "direct_anisotropic_to_target",  # 直接缩放到 240×320, 无填充
    "model_output_normalization": "per_image_min_max_0_1",  # 原生相对量, 非概率
    "adapter_renormalization": "sum_to_1_probability_density",  # 适配层重归一化
    "adapter_output_dtype": "float64",
}

def _viewing_conditions(window_s: int) -> dict[str, Any]:
    """Per-window viewing-condition assumptions (UEyes 1s/3s/7s 真值窗口;
    无 px/dva 类显示条件要求——区别于 DeepGaze MSDB)。"""
    return {
        "viewing_window_s": window_s,
        "window_source": f"UEyes (Jiang et al. 2023) {window_s}s 观看窗口真值; "
        "foveacast-training v0.2.0 按窗口分别训练的三个模型之一",
        "assumption": f"游戏截图的实际观看条件未实测; 以 UEyes {window_s}s 窗口分布作为"
        "工程假设, 不构成对游戏玩家观看行为的断言",
        "display_conditions_required": False,
    }


FOVEACAST_1S_VIEWING_CONDITIONS: dict[str, Any] = _viewing_conditions(1)
FOVEACAST_3S_VIEWING_CONDITIONS: dict[str, Any] = _viewing_conditions(3)
FOVEACAST_7S_VIEWING_CONDITIONS: dict[str, Any] = _viewing_conditions(7)

#: foveacast 无显式 centerbias 输入（冻结规则 → profile.centerbias=None）。
CENTERBIAS_NOTE = (
    "foveacast/MSI-Net 无显式 centerbias 输入; 其中心倾向隐含在训练权重中 "
    "(UEyes 观看分布)。分析侧不得另行叠加先验 (technical-design §5 G0 锁定), "
    "该隐含偏置作为后端已知特性记录于 profile 登记与报告, 而非 centerbias 引用。"
)


@dataclass(frozen=True)
class ProfileRegistration:
    """Registry-internal registration entry (NOT a contract type).

    Wraps the C1 :class:`ResolvedProfile` with C2 registration metadata that
    the frozen contract type does not carry: license status/record, CC-BY-4.0
    attribution obligation, centerbias note and the model-cache subdir.
    """

    profile: ResolvedProfile
    license_status: dict[str, Any]
    license_record: dict[str, Any]
    attribution: tuple[str, ...]
    centerbias_note: str
    cache_subdir: str = "foveacast"


def compute_config_hash(
    *,
    backend_id: str,
    backend_version: str,
    weights: tuple[WeightRef, ...],
    preprocessing: dict[str, Any],
    centerbias: dict[str, Any] | None,
    viewing_conditions: dict[str, Any],
) -> str:
    """Composite hash over ALL registered configuration (contract §3).

    Covers exactly the elements enumerated by C1 ``ResolvedProfile`` docstring:
    backend_id+version、权重清单与哈希、预处理参数、centerbias 引用、观看条件
    假设。Recipe (stable, documented): canonical JSON — ``sort_keys=True``,
    separators ``(",", ":")``, ``ensure_ascii=False`` — UTF-8 encoded, SHA-256,
    bare 64-char lowercase hex (satisfies C1 ``_check_sha256``). Any change to
    any registered element necessarily changes the hash.
    """
    payload = {
        "backend_id": backend_id,
        "backend_version": backend_version,
        "weights": [w.to_dict() for w in weights],
        "preprocessing": dict(preprocessing),
        "centerbias": dict(centerbias) if centerbias is not None else None,
        "viewing_conditions": dict(viewing_conditions),
    }
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_profile_integrity(profile: ResolvedProfile) -> None:
    """Recompute the composite hash and reject runtime-tampered profiles.

    Catches both shallow tampering (``object.__setattr__`` on the frozen
    dataclass) and deep tampering (mutating a preprocessing/viewing dict in
    place). Failure → ``PROFILE_NOT_REGISTERED`` with
    ``details.reason="profile_tampered"`` (exit 3; 对齐 C1 BackendRegistry
    Protocol 文档：未登记或运行时改写 → PROFILE_NOT_REGISTERED)。
    """
    recomputed = compute_config_hash(
        backend_id=profile.backend_id,
        backend_version=profile.backend_version,
        weights=tuple(profile.weights),
        preprocessing=profile.preprocessing,
        centerbias=profile.centerbias,
        viewing_conditions=profile.viewing_conditions,
    )
    if recomputed != profile.config_hash:
        raise ProfileRejectedError(
            f"profile {profile.profile_name!r} failed config-hash integrity check "
            "(runtime modification is forbidden)",
            {
                "profile_name": profile.profile_name,
                "registered_config_hash": profile.config_hash,
                "recomputed_config_hash": recomputed,
            },
            reason="profile_tampered",
        )


def _register_foveacast_window(
    *,
    backend_id: str,
    profile_name: str,
    weight: WeightRef,
    viewing_conditions: dict[str, Any],
) -> ProfileRegistration:
    """Register one foveacast viewing-window profile (1s/3s/7s share the same
    preprocessing contract; only the weight and viewing-window assumption differ)."""
    config_hash = compute_config_hash(
        backend_id=backend_id,
        backend_version=FOVEACAST_3S_BACKEND_VERSION,
        weights=(weight,),
        preprocessing=FOVEACAST_3S_PREPROCESSING,
        centerbias=None,
        viewing_conditions=viewing_conditions,
    )
    profile = ResolvedProfile(
        profile_name=profile_name,
        backend_id=backend_id,
        backend_version=FOVEACAST_3S_BACKEND_VERSION,
        preprocessing=dict(FOVEACAST_3S_PREPROCESSING),
        centerbias=None,
        viewing_conditions=dict(viewing_conditions),
        config_hash=config_hash,
        weights=(weight,),
    )
    profile.validate()  # enforce C1 contract validation at registration time
    return ProfileRegistration(
        profile=profile,
        license_status=dict(FOVEACAST_LICENSE_STATUS),
        license_record=json.loads(json.dumps(LICENSE_RECORD, ensure_ascii=False)),
        attribution=ATTRIBUTION,
        centerbias_note=CENTERBIAS_NOTE,
        cache_subdir="foveacast",
    )


# ---------------------------------------------------------------------------
# DeepGaze IIE（通用域对照基线；G1' 批准 2026-09-11：仅限内部研究评估，
# 不打包不分发。许可缺口 G1/G2/G5 未闭合，如实登记。）
# ---------------------------------------------------------------------------

_DEEPGAZE_RELEASE_BASE = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/"

#: 官方 release（2021）无 digest 公布 → sha256 为下载后本地计算值并在此登记
#: （R3 §2.4 既定策略；PROVENANCE.json 记录同值与口径）。size_bytes 为实测。
DEEPGAZE_IIE_WEIGHT = WeightRef(
    name="deepgaze2e.pth",
    source_url=_DEEPGAZE_RELEASE_BASE + "deepgaze2e.pth",
    sha256="49b65724184f43a2b60426c29c12fd2b743b47ad9eb3435d3dd28e50ea4e2585",
    size_bytes=419676716,
)
DEEPGAZE_CENTERBIAS_WEIGHT = WeightRef(
    name="centerbias_mit1003.npy",
    source_url=_DEEPGAZE_RELEASE_BASE + "centerbias_mit1003.npy",
    sha256="051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9",
    size_bytes=8388688,
)

DEEPGAZE_IIE_LICENSE_STATUS: dict[str, Any] = {
    "code": "未闭合 (仓库无 LICENSE 文件; setup.py MIT 字段被注释; Issue #15 请求商用许可 "
    "open、0 评论、维护者未回应——默认著作权保留所有权利)",
    "weights": "未闭合 (release v1.0.0 页与 deepgaze2e.pth/centerbias 文件本身均无许可声明; "
    "pth 内含 ShapeNetC 骨干权重, 其 bitbucket 来源条款未声明=G2)",
    "gaps": ["G1", "G2", "G5"],
    "cleared_for": "internal-eval",
}

DEEPGAZE_LICENSE_RECORD: dict[str, Any] = {
    "status": DEEPGAZE_IIE_LICENSE_STATUS,
    "evidence": {
        "code": "jsDelivr 全量文件清单无许可文件 + setup.py raw 原文 MIT 字段注释状态 (R1 §C3 已验证); "
        "Issue #15 (2023-10-11) open、0 评论 (GitHub API 经代理转发, R1 实时核查)",
        "weights": "release 资产页与文件本身无声明 (R1 §C3 已验证)",
        "training_data": "IIE: MIT1003; MSDB 另含 CAT2000/COCO-FreeView/Daemons/Figrim——各数据集条款"
        "未逐一核实 (G5, R1 时间盒收口)",
        "backbone": "IIE 4 骨干: ShapeNetC (bitbucket 权重条款未声明=G2), vendored efficientnet_pytorch "
        "(上游 Apache-2.0), torchvision densenet201/resnext50 (框架 BSD-3, ImageNet 权重残留风险=G8 类比)",
        "source_pin": DEEPGAZE_SOURCE_PIN,
    },
    "gaps_detail": {
        "G1": "DeepGaze 代码与权重无任何许可声明; 默认保留所有权利 → 不得打包、不得再分发; "
        "内部研究评估经一级总控 G1' 批准 (2026-09-11)",
        "G2": "ShapeNetC 骨干权重 (texture-vs-shape bitbucket) 条款未声明; 其权重已含于 deepgaze2e.pth",
        "G5": "MIT1003/CAT2000/COCO-FreeView/Daemons/Figrim 训练数据条款未逐一核实",
    },
    "obligation": "仅限内部研究评估 (G1' 批准 2026-09-11); 禁止打包/分发/商用; 代码与权重仅存在于本地"
    "隔离环境 (.venv + model-cache, 均 git 忽略目录), 不 fork 上游入仓; 使用记录须携带来源 pin "
    + DEEPGAZE_SOURCE_PIN + " 与学术引用 (Linardos et al. 2021)。",
    "attribution": [
        "Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). Calibrated prediction in and "
        "out-of-domain for state-of-the-art saliency modeling. arXiv:2105.12441. "
        "(学术引用——DeepGaze 无许可声明, 引用非法律义务而是学术规范)",
    ],
    "cleared_for": "internal-eval",
}

DEEPGAZE_ATTRIBUTION: tuple[str, ...] = tuple(DEEPGAZE_LICENSE_RECORD["attribution"])

#: 预处理参数（R1 已验证源码/README；均值归一化在模型内部 Normalizer 完成，
#: 适配层传 [0,255]；输出原生 log_density，适配层不做 exp/重归一化——
#: C1 metrics 按 technical-design §7 转换）。
DEEPGAZE_IIE_PREPROCESSING: dict[str, Any] = {
    "target_long_side": 1024,  # MIT1003 训练分布尺度（35 px/dva 实验假设的一部分）
    "resize_policy": "aspect_preserving_downscale_only",  # 只降不升，保留宽高比
    "resize_filter": "pillow_bilinear",
    "input_layout": "NCHW",
    "channel_order": "RGB",
    "input_dtype": "float32",
    "value_range": [0.0, 255.0],
    "mean_subtraction": "in_model_normalizer",  # Normalizer: /255 + ImageNet mean/std（模型内部）
    "centerbias_zoom": {"order": 0, "mode": "nearest"},  # README 官方口径
    "centerbias_renormalization": "subtract_logsumexp",
    "output_semantics": "log_density_native_no_adapter_renormalization",
    "adapter_output_dtype": "float64",
    "pixel_per_degree_assumption": 35.0,
}

DEEPGAZE_IIE_VIEWING_CONDITIONS: dict[str, Any] = {
    "pixel_per_degree": 35.0,
    "assumption": "MIT1003 采集条件 (35 px/dva, 图像长边约 1024) 作为固定实验配置=工程实验假设"
    " (technical-design §5: 固定实验配置可以用于工程比较, 但必须标记为实验假设); "
    "游戏截图的真实显示尺寸/观看距离未实测, 不得从截图猜测",
    "source": "MIT1003 via DeepGaze README + centerbias_mit1003.npy 模板 (release v1.0.0)",
    "display_conditions_required": False,  # IIE 不在运行时消费 ppd（区别于 MSDB 必填参数）
}

#: 显式 centerbias 引用（契约 §3: 可选，含来源与哈希——DeepGaze IIE 必须显式提供）。
DEEPGAZE_IIE_CENTERBIAS: dict[str, Any] = {
    "name": "centerbias_mit1003.npy",
    "source": "MIT1003 先验模板 (DeepGaze release v1.0.0, 2021-06-01)",
    "source_url": _DEEPGAZE_RELEASE_BASE + "centerbias_mit1003.npy",
    "sha256": DEEPGAZE_CENTERBIAS_WEIGHT.sha256,
    "template_shape": [1024, 1024],
    "semantics": "log_density",
    "usage": "zoom(order=0, mode=nearest) 到推理尺寸后 -= logsumexp 重归一 (README 官方口径)",
    "assumption": "MIT1003 中心偏置不代表游戏玩家真实注视习惯 (technical-design §5: 不得如此声称); "
    "A/B 对比必须使用同一先验版本",
}

_DEEPGAZE_CENTERBIAS_NOTE = (
    "DeepGaze IIE 接受显式 centerbias log-density 输入; 本项目使用版本化登记的 MIT1003 模板"
    " (来源+哈希见 profile.centerbias)。不得把 MIT1003 的中心偏置称为游戏玩家的真实习惯。"
)


def _register_deepgaze_iie_v1() -> ProfileRegistration:
    config_hash = compute_config_hash(
        backend_id=DEEPGAZE_IIE_BACKEND_ID,
        backend_version=DEEPGAZE_IIE_BACKEND_VERSION,
        weights=(DEEPGAZE_IIE_WEIGHT, DEEPGAZE_CENTERBIAS_WEIGHT),
        preprocessing=DEEPGAZE_IIE_PREPROCESSING,
        centerbias=DEEPGAZE_IIE_CENTERBIAS,
        viewing_conditions=DEEPGAZE_IIE_VIEWING_CONDITIONS,
    )
    profile = ResolvedProfile(
        profile_name=DEEPGAZE_IIE_PROFILE,
        backend_id=DEEPGAZE_IIE_BACKEND_ID,
        backend_version=DEEPGAZE_IIE_BACKEND_VERSION,
        preprocessing=dict(DEEPGAZE_IIE_PREPROCESSING),
        centerbias=dict(DEEPGAZE_IIE_CENTERBIAS),
        viewing_conditions=dict(DEEPGAZE_IIE_VIEWING_CONDITIONS),
        config_hash=config_hash,
        weights=(DEEPGAZE_IIE_WEIGHT, DEEPGAZE_CENTERBIAS_WEIGHT),
    )
    profile.validate()
    return ProfileRegistration(
        profile=profile,
        license_status=dict(DEEPGAZE_IIE_LICENSE_STATUS),
        license_record=json.loads(json.dumps(DEEPGAZE_LICENSE_RECORD, ensure_ascii=False)),
        attribution=DEEPGAZE_ATTRIBUTION,
        centerbias_note=_DEEPGAZE_CENTERBIAS_NOTE,
        cache_subdir="deepgaze",
    )


#: Registration table. Populated at import; there is no public mutation API —
#: 执行时不可改写 (contract §3). Adding a profile is a code change that must
#: ship a new ``-v<N>`` name with a fresh config hash. 3s = 主用窗口（G0 首登记）；
#: 1s/7s = 窗口敏感性分析（G2 批准，许可字段同 3s）；deepgaze-iie-v1 =
#: 通用域对照基线（G1' 批准：仅限内部研究评估，不打包不分发）。
_REGISTRATIONS: dict[str, ProfileRegistration] = {
    FOVEACAST_3S_PROFILE: _register_foveacast_window(
        backend_id=FOVEACAST_3S_BACKEND_ID,
        profile_name=FOVEACAST_3S_PROFILE,
        weight=FOVEACAST_3S_WEIGHT,
        viewing_conditions=FOVEACAST_3S_VIEWING_CONDITIONS,
    ),
    FOVEACAST_1S_PROFILE: _register_foveacast_window(
        backend_id=FOVEACAST_1S_BACKEND_ID,
        profile_name=FOVEACAST_1S_PROFILE,
        weight=FOVEACAST_1S_WEIGHT,
        viewing_conditions=FOVEACAST_1S_VIEWING_CONDITIONS,
    ),
    FOVEACAST_7S_PROFILE: _register_foveacast_window(
        backend_id=FOVEACAST_7S_BACKEND_ID,
        profile_name=FOVEACAST_7S_PROFILE,
        weight=FOVEACAST_7S_WEIGHT,
        viewing_conditions=FOVEACAST_7S_VIEWING_CONDITIONS,
    ),
    DEEPGAZE_IIE_PROFILE: _register_deepgaze_iie_v1(),
}


def list_profiles() -> tuple[str, ...]:
    """All registered profile names (read-only; BackendRegistry Protocol)."""
    return tuple(_REGISTRATIONS)


def registration_for(profile_name: str) -> ProfileRegistration:
    """Full registration entry (license/attribution metadata) by name."""
    if not isinstance(profile_name, str) or profile_name not in _REGISTRATIONS:
        raise ProfileRejectedError(
            f"backend profile not registered: {profile_name!r}",
            {"requested": repr(profile_name), "registered": list(_REGISTRATIONS)},
            reason="not_registered",
        )
    return _REGISTRATIONS[profile_name]


def resolve_profile(profile_name: str) -> ResolvedProfile:
    """Resolve a registered profile name (BackendRegistry Protocol; C1 cli).

    Raises:
        ProfileRejectedError: unknown/empty name → ``PROFILE_NOT_REGISTERED``
            (exit 3; never silently substituted). Returned profile passes the
            config-hash integrity check.
    """
    registration = registration_for(profile_name)
    profile = registration.profile
    verify_profile_integrity(profile)
    return profile


def profile_for_backend(backend_id: str) -> ResolvedProfile:
    """First registered profile for a backend id (used by describe())."""
    for registration in _REGISTRATIONS.values():
        if registration.profile.backend_id == backend_id:
            verify_profile_integrity(registration.profile)
            return registration.profile
    raise ProfileRejectedError(
        f"no registered profile for backend_id {backend_id!r}",
        {"backend_id": backend_id, "registered": list(_REGISTRATIONS)},
        reason="backend_not_registered",
    )


def license_record(profile_name: str = FOVEACAST_3S_PROFILE) -> dict[str, Any]:
    """License verification record for doctor/report consumption (no credentials)."""
    return registration_for(profile_name).license_record


def get_backend(profile: ResolvedProfile) -> Backend:
    """Instantiate the backend for a resolved profile (BackendRegistry Protocol).

    Offline weight verification happens here (missing file / sha256 mismatch →
    ``MODEL_NOT_READY``, exit 3). Downloading is a separate explicit step
    (``python -m ui_attention.backends``), never implicit.

    Raises:
        ProfileRejectedError: tampered profile or unknown backend_id.
        WeightNotReadyError: weights missing or hash/size mismatch.
    """
    verify_profile_integrity(profile)
    registration = registration_for(profile.profile_name)
    if (
        profile.backend_id not in FOVEACAST_BACKEND_IDS
        and profile.backend_id != DEEPGAZE_IIE_BACKEND_ID
    ):
        raise ProfileRejectedError(
            f"no backend implementation registered for backend_id {profile.backend_id!r}",
            {
                "backend_id": profile.backend_id,
                "supported": [*FOVEACAST_BACKEND_IDS, DEEPGAZE_IIE_BACKEND_ID],
            },
            reason="backend_mismatch",
        )
    from .weights import cache_root, verify_cached_weight

    cache_dir = cache_root() / registration.cache_subdir
    for ref in profile.weights:
        verify_cached_weight(ref, cache_dir)  # 缺失/哈希不符 → MODEL_NOT_READY (exit 3)
    if profile.backend_id in FOVEACAST_BACKEND_IDS:
        from .onnx_foveacast import FoveacastOnnxBackend

        return FoveacastOnnxBackend(backend_id=profile.backend_id, cache_dir=cache_dir)
    from .deepgaze_iie import DeepGazeIIEBackend

    return DeepGazeIIEBackend(cache_dir=cache_dir)


def doctor_report() -> dict[str, Any]:
    """Structured doctor result (BackendRegistry Protocol; C1 cli consumes).

    Delegates to :mod:`ui_attention.backends.doctor`; import is function-level
    to avoid an import cycle (doctor imports registry).
    """
    from .doctor import run_doctor

    return run_doctor().to_dict()
