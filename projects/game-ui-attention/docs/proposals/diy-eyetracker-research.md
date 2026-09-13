# DIY 眼动仪方案研究（A3）

- 文档日期：2026-09-13
- 起草子任务：A3（眼动仪 DIY 方案研究线），向 DHS 二级总控（L2）汇报
- 状态：研究产出（供 L2 决策，未获立项/采购授权）
- 性质：**可公开脱敏文档**——不含内部截图、凭据、个人身份信息、本机绝对路径；价格一律标注"需实购验证/需询价"；SGAME 界面仅以界面族类别（结算/商店/任务/弹窗）指代。
- 证据分级：`已核实` = 本任务 2026-09-13 直接抓取一手页面/API 原文（GitHub API、仓库 LICENSE 全文、项目页）；`推断` = 由证据合理推导但未实测；`未核实` = 未取得一手来源（附原因）。设备精度/价格为厂家标称、文献值或社区实测值转述，正式采购前以规格书/实测为准。
- **单文件独占产出**：本文档只写 `docs/proposals/diy-eyetracker-research.md`；不下载二进制/数据集/硬件设计文件；不写 Git；不改其他文件。

---

## 0. 摘要与建议速览（结论先行）

1. **一句话结论**：**自建纯摄像头方案（WebGazer / MediaPipe iris / L2CS-Net 系）精度 1°–10°，不足以支撑 NSS/IG/sAUC 级一致性验证，只够"粗粒度 AOI"或"采集-重建-指标全链预研"；DIY 家族里唯一精度可能贴近正式验证下限的是**二手/存量商业硬件复用（EyeTribe ~1°、Tobii 4C ~0.5–1°）**，但二者均已停产、开源驱动生态碎片化且多已停维护，只能定位为 **Gate1 前协议预研 pilot**，不能作为正式验证设备。
2. **推荐策略（组合）**：采纳"**DIY 作 Gate1 前的协议预研 pilot**"——选**二手 Tobii 4C 或 EyeTribe**（若可低成本获得）或**L2CS-Net 纯软件 webcam**（零硬件成本、最快跑通），目标=验证"采集→重建→指标→bootstrap"全链跑通、标定/质控流程、设备适配器接口形态；**正式一致性数字（NSS/IG/sAUC）仍以 A2 首选 Tobii Pro Fusion 250（降级 Eye Tracker 5）为准**，商业设备到货后复测。DIY 数据与商业设备数据**严格分表、不混表**（沿用协议 §8.1 可比性清单）。
3. **致命风险（详见 §6）**：精度不足导致一致性数字不可解释；标定失败率高；社区项目停维护（PyTribe 2016 停更、Tobii 4C 开源驱动 2020 停更）；EyeTrackVR 为 VR 场景 + 非商业 copyleft 自定义许可，不适配"看屏幕"任务且许可受限。
4. **口径对齐声明（§2）**：本文只引用、不重述 A2/协议口径；对 A2 未展开的 Pupil 软件许可做了补充（LGPL-3.0，已核实），非冲突，属细化。

---

## 1. 目标与范围

- **目标**：盘点"用开源硬件/软件/现成摄像头自建低成本眼动采集"的可行路线，评估其能否支撑本项目的验证需求——NSS/IG/sAUC 级一致性验证，或退而求其次的粗粒度 AOI 一致性。
- **范围边界**：仅 web 调研 + 写本单文件；不下载任何二进制/数据集/硬件设计文件；价格仅区间 + 需实购验证；许可仅作风险提示，不构成法务结论。
- **验证需求口径**（引用 [benchmark-protocol.md](../research/benchmark-protocol.md) §6/§7、§2.1，[eyetracker-validation-proposal.md](eyetracker-validation-proposal.md) §2/§4，不重述）：真值注视点重建 = 有效标志（对齐 Gazepoint `BPOGV` 语义）+ 观看窗口时长前缀截断 + 屏幕归一化坐标 → 显示分辨率 → letterbox 逆变换 → 图像像素坐标；指标 = IG/NSS/CC/sAUC/AUC-Judd/KL；真值模糊 σ=1°；不确定性 = 图像级 bootstrap。

---

## 2. 与既有口径的对齐声明

本文**只引用、不重述**既有口径，冲突处显式标注。

| 既有口径 | 出处 | 本文用法 |
| --- | --- | --- |
| WebGazer 精度 1°–5°、GPLv3（<$100万可 LGPLv3）、官方维护 2026-02-24 结束 | A2 §2.1/§2.2 | 直接引用，不重复论述（§3.1 仅列接入点） |
| Tobii Pro Fusion 250（0.3° 标称/250 Hz）、Eye Tracker 5（135 Hz、精度未公布推断 0.4–0.9°） | A2 §2.1 | 三方对照表基准（§4） |
| Pupil Labs Core/Neon 为穿戴式、开源软件、需/免校准 | A2 §2.1 | 仅就"开源软件栈可否复用"展开（§3.2），硬件采购不属 DIY 范畴 |
| 注视重建口径（BPOGV 等价有效标志/时长窗/letterbox 逆变换/越界丢弃） | protocol §2.1/§6.0、`ueyes_driver.py`、`groundtruth.py` | DIY 数据流必须对齐（§5 差异点表） |
| FixationSet 契约（points=(n,2) 图像像素 (x,y)、weights、n_dropped） | `groundtruth.py` | DIY 适配器输出目标（§5） |
| 可比性清单（§8.1 八项） | protocol §8.1 | DIY 数据与商业设备数据分表依据（§4/§7） |

**补充/细化（非冲突，已在 A2 基础上新增一手核实）**：A2 对 Pupil 仅称"开源软件"；本任务经 GitHub API 核实 `pupil-labs/pupil` 主仓许可为 **LGPL-3.0**（1732★/705 fork/最近推送 2026-08-31）。此为细化 A2 口径，不构成冲突（§3.2）。

---

## 3. DIY 路线盘点（≥5 条）

> 访问日期均为 2026-09-13。精度（°）分级标注；价格为区间，**需实购验证/需询价**；采样率为标称或推断。

### 3.1 纯软件 webcam 视线估计（路线 a）

**原理**：普通 RGB webcam + 深度学习/几何模型，从人脸/眼睛图像估计注视方向，无 IR 硬件。

| 子项 | MediaPipe FaceMesh iris | WebGazer.js | L2CS-Net | ETH-XGaze 系 / Gaze360 系 |
| --- | --- | --- | --- | --- |
| 精度（°） | **无官方精度**；社区报告约 3–5°（irid 偏移近似视线）[推断/未核实] | **1°–5°**（A2 引用，此处不重述）[论文口径] | 文献值 MPIIFaceGaze ≈3.92°、Gaze360 ≈10.41°（mean angular error）[文献值，PDF 未核实] | 该二者主要为**数据集/训练基准**；其上训练的模型 3–10° 量级[推断] |
| 采样率 | webcam 帧率 30–60 Hz | 30–60 Hz | 30–60 Hz（受推理速度限制，CPU 实时性差） | — |
| BOM（人民币量级） | 软件 ¥0 + 现成 webcam ¥0–200（如需另购 720p/1080p）[需实购验证] | ¥0 | ¥0 + webcam | ¥0 + webcam |
| 搭建工时（人天） | 1–2（调 Face Landmarker + iris→视线映射 + 标定） | 0.5–1（浏览器 demo，改数据导出） | 2–4（装 PyTorch + 权重 + face 检测 + 标定 + 数据流） | — |
| 标定与失败模式 | 需个人标定（视点头→屏幕点多项式回归）；头部移动/光照敏感；眼镜/斜视易失败 | 自校准（点击/光标交互），精度随头部移动退化 | 需逐人标定；无 IR 下低照度/眼镜失败率上升 | — |
| Python 数据流 → FixationSet | MediaPipe Python `FaceLandmarker` 输出归一化 iris 中心 (x,y) → 视线映射 → 屏幕归一化 → letterbox 逆变换（**需自写映射层**） | JS 端输出归一化屏幕坐标，需自建 JS→日志 CSV 桥（无原生 Python） | `l2cs.Pipeline.step(frame)` 输出 pitch/yaw（3D 视线角）→ 屏幕交点投影（**需自写投影+标定**） | 同左 |
| 许可 | Apache-2.0（MediaPipe 本体）[推断，未见 LICENSE 原文] | GPLv3/LGPLv3（A2） | **MIT**（GitHub API 已核实） | ETH-XGaze 数据集 **CC BY-NC-SA 4.0（非商用）**（项目页已核实） |
| 隐私 | 全本地，可满足保密红线 | 客户端本地（A2 已确认加分项） | 全本地 | 全本地 |
| 活跃度 | MediaPipe 持续维护（官方） | 官方维护已停（A2） | **527★/114 fork，最近推送 2024-02-02（停滞但未归档）**（API 已核实） | ETH-XGaze 数据按申请获取；模型生态活跃 |

**评估**：精度 1°–10°，**全部无法支撑 NSS/IG 级**；L2CS-Net（MIT、有现成 `l2cs` Python 包）是"零硬件成本最快跑通全链"的最佳候选，但精度下限约 3°，只适合**协议全链预研 + 粗 AOI**。MediaPipe iris 只给几何近似视线，不是研究级 gaze，标注"无官方精度"。

### 3.2 头戴式 DIY 硬件（路线 b）

**原理**：头戴支架 + 全局快门/IR 相机 + IR LED，采集眼部近景（瞳孔/角膜反射），复用开源软件栈。

| 子项 | EyeTrackVR | Pupil Labs 开源栈复用 |
| --- | --- | --- |
| 仓库/活跃度 | `EyeTrackVR/EyeTrackVR`：**1143★/93 fork，最近推送 2026-09-07（非常活跃）**；Python（GitHub API 已核实） | `pupil-labs/pupil`：**1732★/705 fork，最近推送 2026-08-31（活跃）**；Python（GitHub API 已核实） |
| 目标场景 | **VR 头显内眼动**（topics：chilloutvr/neosvr/vrchat）；**非"看桌面屏幕"场景**，屏幕映射需自行改造 | Pupil Core 穿戴式硬件配套软件；硬件本体**商业销售、非 DIY**（复用其开源软件需自备眼相机硬件） |
| 硬件 BOM（人民币量级） | ESP32-S3 开发板 ×2 + OV2640/OV9281 相机 ×2 + IR LED + 3D 打印支架 ≈ **¥200–500**（不含 3D 打印工时）[需实购验证] | 无官方 DIY BOM；等效自建需全局快门相机 + IR 照明 ≈ ¥300–800 [需实购验证] |
| 采样率 | 相机依赖，工程实践 60–120 Hz [推断] | 眼相机 120–200 Hz（Pupil Core 硬件口径，A2） |
| 精度（°） | 未正式公布；VR 场景社区口径 1–3° [推断/未核实] | Core 标称 ≈0.6°（A2，需规格书） |
| 搭建工时（人天） | 5–10（硬件 + 固件 + 改装屏幕映射 + 标定） | 5–8（自建硬件 + 跑 Pupil 软件 + 标定 + 数据流） |
| 许可 | **自定义 "Babble Software Distribution License 1.0"（2025-01）：非商业 + copyleft**（LICENSE 全文已核实）：禁止商用、禁止集成进收费硬件、衍生作品须同许可开源公开源码 | **LGPL-3.0**（主仓 API 已核实；个别组件历史为 GPLv3，见 §6 许可注记） |
| 隐私 | 全本地 | 全本地 |
| 数据流 → FixationSet | 输出 VR 内视线，需自研"VR 注视→屏幕交点"映射层（工作量即主要风险） | Pupil Capture/Service 网络 API 输出场景相机归一化注视；**穿戴式看屏幕需 Reference Image Mapper 映射**（同 A2 Neon 多一步） |

**评估**：EyeTrackVR 活跃且社区大，但**目标场景是 VR，不是桌面看屏幕**，且许可为**非商业 copyleft 自定义许可**——用于商业游戏公司内部研究存在合规歧义（内部科研大概率非商业，但"商业实体使用 + 衍生须开源"需法务确认）。Pupil 软件栈许可宽松（LGPL-3.0）但硬件本体商业、DIY 需自建。二者精度（1–3° 推断）均难达 NSS/IG 级。

### 3.3 屏幕式 DIY（显示器下置相机 + IR 照明）（路线 c）

**原理**：相机置于显示器下缘，配 940nm IR LED 照亮眼睛，摄像头改装（移除 IR-cut + IR-pass）后做瞳孔检测 + 角膜反射，标定映射到屏幕。

- 开源项目：OpenGaze（`github.com/opengaze/opengaze`、`opengaze.io`）及大量树莓派/OpenCV 教程（社区级，非统一成熟产品）。
- 典型硬件：Raspberry Pi 4/5 + Pi Camera Module 3 NoIR（或改装 Pi Camera）+ 2–4 颗 940nm IR LED + 支架。
- 精度：**无统一标称，社区实测约 1–3°**（光照/头位/镜片敏感）[推断，社区教程口径]。
- 采样率：Pi Camera 60–120 Hz（分辨率换帧率）[推断]。
- BOM：**≈¥600–1000**（Pi 5 ≈¥400–600 + NoIR Camera ≈¥200–300 + IR LED/支架 ≈¥30–100）[需实购验证]。
- 工时：**4–8 人天**（硬件组装 + IR 改装 + OpenCV 瞳孔/角膜反射检测 + 标定 + 数据流）。
- 标定：5/9 点标定 + 漂移校正；失败模式=眼镜反光、环境 IR 干扰、头部移动、低对比度。
- 数据流：Pi 端（或回传 PC 端 OpenCV/Python）输出屏幕归一化注视 → letterbox 逆变换 → FixationSet（**标定矩阵需自写**）。
- 许可：OpenGaze 及教程多为个人项目/MIT/无明确许可 [推断，未见 LICENSE]；OpenCV 为 Apache-2.0。
- 隐私：全本地。
- 活跃度：无单一权威项目，生态碎片化；OpenGaze 仓库活跃度低 [推断]。

**评估**：这条是"纯自建屏幕式"里精度上限最高的路线（IR 照明 + 近距离眼相机），但仍是 1–3° 推断，且无成熟工程、标定复杂度最高，**回报/工时比最低**，仅作技术储备，不作为首选 DIY。

### 3.4 二手/存量商业硬件复用（路线 d）

**原理**：复用已停产商业眼动仪 + 开源驱动/SDK，硬件无需自建。

| 子项 | EyeTribe（已停产） | Tobii 4C / Eye Tracker 4（已停产） |
| --- | --- | --- |
| 停产状态 | The Eye Tribe 2016 年被 Oculus/Facebook 收购后停产；定位低成本开发套件 [推断，业界通行] | Tobii 4C/ET4 已停产（消费级，后继 Eye Tracker 5）；4C 采样 90 Hz |
| 精度（°） | 标称 0.5–1°（当年口径）；社区实测 ≈1° [推断/需实购验证] | 消费级未正式公布，推断 0.5–1°（略优于纯软件）[推断] |
| 采样率 | 30 Hz（基础）/ 60 Hz（开发模式） | 90 Hz |
| BOM（二手） | 二手 ≈**$50–150**（eBay/闲鱼，波动大）[需实购验证] | 二手 ≈**$50–100** [需实购验证] |
| 工时（人天） | 2–3（装 PyTribe + 旧 SDK + adapter） | 3–5（驱动/SDK 兼容 + adapter） |
| SDK/驱动 | **PyTribe**（`esdalmaijer/PyTribe`：GPL-3.0，28★，**最近推送 2016-07-22，停维护**，API 已核实）；libEyeTribe（C++） | 官方消费 SDK=Tobii Stream Engine（免费开发者）；开源驱动碎片化：`Eitol/tobii_eye_tracker_linux_installer`（177★，C，**2020-01 停更，无许可声明**）、`DigitalNatureGroup/TobiiEyeTracker.py`（17★，2022 停更）、`GazeOSC`（MIT，2022 停更）（API 已核实）；**历史痛点：4C 取原始注视需 Tobii Gaze SDK 授权/逆向**（社区 repo 说明已核实） |
| 许可 | PyTribe GPL-3.0（**GPL 传染风险**：链接/分发需 GPL 兼容；内部工具不入仓可规避，见 §6） | 开源驱动多无明确许可或 MIT；官方 Stream Engine 为专有 EULA |
| 隐私 | 全本地 | 全本地 |
| 活跃度 | 停维护（2016 后无实质更新） | 开源生态整体停更（2019–2022）；官方 Stream Engine 仍随 ET5 维护 |
| 数据流 → FixationSet | PyTribe 回调输出屏幕归一化注视 (x,y)+时间戳+有效标志 → letterbox 逆变换 → FixationSet（**最贴近 Gazepoint 范式，适配成本最低**） | Stream Engine/开源驱动输出归一化注视+有效标志 → 同左 |

**评估**：**DIY 家族中精度上限最高（0.5–1°）、数据流最贴近既有 Gazepoint 范式**，是唯一"可能摸到正式验证下限"的 DIY 类路线；但停产 + 驱动停维护 + 二手硬件无质保是硬伤，只能作预研 pilot。另注：`tobii-glasses`（如 `ddetommaso/TobiiGlassesPyController`，GPL-3.0，62★，2024-08 停更）指 **Tobii Pro Glasses 2/3 穿戴式眼镜**，非屏幕式 4C，与本项目"看屏幕截图"任务形态不符，不采纳。

### 3.5 手机前置摄像头方案（路线 e）

**原理**：手机前置摄像头 + ML 视线估计 app（无前置 IR）。

- 精度：**无 IR、无角膜反射，仅外观法**，社区 app 口径 2–5°（弱于桌面 webcam 同法）[推断/未核实]。
- 采样率：前置摄像头 30–60 Hz。
- BOM：¥0（现有手机）+ 支架（需固定头位）≈¥20–50 [需实购验证]。
- 工时：1–2 人天（选 app + 导出日志 + 适配）。
- 标定：app 内标定；失败模式=低照度、头动、前置摄像头无 IR 下瞳孔对比度差。
- 数据流：app 导出 CSV（归一化注视）→ letterbox 逆变换 → FixationSet；**app 数据格式各家私有、许可多为专有/闭源**，导出自由度差。
- 许可：主流 app 闭源 + 服务端处理（**隐私红线风险**：部分 app 上传视频，违反保密红线，须选本地-only app）。
- 活跃度：app 生态商业驱动，个别开源（如 GazeRecorder 等 [推断]），但无统一标准。

**评估**：精度最差、隐私风险最高、数据导出最不可控，**不推荐**；仅在"零预算 + 仅演示粗 AOI"时考虑本地-only app。

---

## 4. 三方对照表与"能否支撑 NSS/IG 级"判断

### 4.1 三方对照表（DIY 最优 vs Eye Tracker 5 vs Fusion 250）

> 价格区间，需实购验证/需询价；精度为标称/推断，正式以规格书/实测为准。

| 维度 | DIY 最优（二手 EyeTribe / Tobii 4C 存量复用） | Tobii Eye Tracker 5（消费级） | Tobii Pro Fusion 250（研究级） |
| --- | --- | --- | --- |
| 精度（°） | **0.5–1°**（标称/推断，需实测校准验收）[推断] | **未正式公布，推断 0.4–0.9°**（A2） | **约 0.3°**（标称，需规格书）（A2） |
| 采样率 | 30/60 Hz（EyeTribe）或 90 Hz（4C） | 135 Hz | 250 Hz |
| 成本（人民币量级） | 二手 **¥400–1100** + 适配工时 [需实购验证] | 约 **¥1400**（$195，A2）[需询价] | **数万元级**（A2）[需询价] |
| 搭建/集成工时（人天） | 2–5（驱动兼容 + 标定 + adapter，**风险高**） | 1–3（官方 SDK 成熟） | 2–4（官方 Python SDK 成熟） |
| 维护 | **停维护**：PyTribe 2016 停更、4C 开源驱动 2020 停更；二手无质保 | 官方 Stream Engine 在维护；硬件在售 | 官方 SDK 成熟、研究生态在维护 |
| 校准研究属性 | 自建标定 + 漂移校正（无官方研究校准） | 自带简易校准，研究级校准能力受限（A2） | 标准化点校准（`ScreenBasedCalibration`） |
| 数据流 → FixationSet 适配成本 | **最低**（归一化注视范式最贴近 Gazepoint） | 低（归一化注视 + 有效标志） | 低 |
| **能否支撑 NSS/IG/sAUC 级验证** | **边际/不推荐**：σ_e≈0.5–1°，见 §4.2 判据 | **可（边界）**：σ_e≈0.4–0.9°，结论标注"粗验证"置信度（A2） | **可**：σ_e≈0.3° < 0.5° 阈值 |
| 结论定位 | **Gate1 前协议预研 pilot**（跑通全链 + 粗 AOI 参照） | 正式验证**降级设备**（粗验证） | 正式验证**首选设备** |

### 4.2 "能否支撑 NSS/IG 级"的判断依据（文献或推断）

本项目协议真值模糊 σ=1°（UEyes σ=40px@1920×1200 ≈1°，protocol §2.4/§6.0）；NSS 在**离散注视点**上采样 z-score 化显著图，IG 在注视点处取 log2(S/B)，sAUC 比较注视点 vs 打乱非注视点的显著值——三者都**以注视点精确坐标**为输入。

**判断依据（推断，非单一文献阈值）**：若注视点存在各向同性高斯误差 σ_e，等效于把显著图 S 与误差核 N(0,σ_e) 卷积后再采样；误差引入的有效模糊 σ_eff = √(σ² + σ_e²)（σ=1°）。信息类指标 IG 随 S 被额外平滑而单调下降，当 σ_e 接近/超过显著特征尺度（游戏 UI 元素典型 1–3°）时，IG/NSS 退化为"注视点 ±1–2° 邻域内显著性是否偏高"的粗粒度判别，与粗 AOI 无本质区别；sAUC 的注视/非注视可分性同步下降、向 0.5 收缩。

由此给出的**经验阈值（推断，冻结前请 L1/L2 复核）**：

| σ_e（°） | 有效模糊 σ_eff（σ=1°） | 判断 |
| --- | --- | --- |
| ≤ 0.5° | ≤ 1.12°（+12%） | 噪声对 σ=1° 真值影响温和，**可支撑 NSS/IG/sAUC** |
| 0.5–1° | 1.12–1.41°（+12%–41%） | **边际**：一致性数字可能被噪声部分主导，只作"方向性/粗验证"，须如实写进 limitations |
| ≥ 2° | ≥ 2.24°（+124%） | 细尺度显著特征被抹除，**仅粗 AOI**，一致性数字不可解释 |

注：该阈值为**本项目协议 σ=1° 口径下的推导**（已核实协议 σ 取值；σ_e→σ_eff 关系为几何/信号处理推导，未取到"注视噪声 vs 显著指标"的单一文献阈值原文，故标 `推断`）。不同项目若改 σ 或改指标集合，阈值需重估。

---

## 5. 与既有管线衔接注记（DIY 采集日志 → 设备适配器）

A2 §4.1 已列"设备适配器"需求（日志→`FixationSet`）。DIY 各路线数据流须对齐 [benchmark-protocol.md §2.1/§6.0](../research/benchmark-protocol.md) 与 `ueyes_driver.py`/`groundtruth.py` 的 `FixationSet` 契约：`points=(n,2)` 图像像素 `(x,y)`、`weights=(n,)`（计数=1/时长=注视时长）、`n_dropped_out_of_bounds`。

**DIY 日志 vs Gazepoint 日志的关键差异点（适配器必须处理）**：

| 差异点 | Gazepoint（UEyes 范式，`ueyes_driver.py` 已支持） | DIY 各路线典型形态 | 适配器动作 |
| --- | --- | --- | --- |
| 坐标系 | `BPOGX/BPOGY` 屏幕归一化 0–1，原点左上 | EyeTribe/PyTribe/4C 亦为归一化 0–1（最接近）；L2CS 输出 **pitch/yaw 视线角**（非屏幕坐标）；MediaPipe 输出 **归一化 iris 位置**（非注视点）；EyeTrackVR 输出 VR 内视线 | 归一化坐标路线直接复用 letterbox 逆变换；**视线角/iris 路线需新增"标定映射层"**（视线角→屏幕交点；iris 偏移→屏幕点多项式回归），映射层参数须版本化入 config 哈希 |
| 时间戳 | `TIME` 秒（会话内），窗口=时长前缀截断 | EyeTribe/4C 提供系统时间戳（ms）；L2CS/MediaPipe 需**自记 frame 到达时间戳** | 统一转会话内秒，维护"媒体名↔trial"对齐；DIY 自记时间戳需标注时钟源与精度 |
| 有效标志 | `BPOGV`（1=有效） | EyeTribe `state`/`avg` 标志；4C validity flag；**L2CS/MediaPipe/WebGazer 无原生"注视有效性"标志** | 需自定等价标志（如置信度阈值、视线角在屏幕内、眨眼排除），并在 config 记录判定规则 |
| 注视 vs 原始采样 | Gazepoint 直接给注视点（含 `FPOGD` 时长） | DIY 多输出**原始帧采样点**，非注视事件 | 需自写**注视聚类（I-VT/I-DT）**生成注视 + 时长；聚类参数（速度阈值/最小时长）版本化入 config，**否则"时长加权"口径不可比** |
| 采样率 | 设备原生 | webcam 30–60 Hz，帧间隔抖动大 | 有效采样率入审计产物（A2 §10 已裁决）；必要时按时间戳重对齐 |

**结论**：二手 EyeTribe/4C 的归一化注视日志最贴近 Gazepoint 范式，适配器改动最小（仅列名/有效标志映射）；L2CS/MediaPipe 路线需**额外自研"视线角/iris→屏幕交点"标定层 + 注视聚类层**，这两层正是 DIY 精度的主要误差源，且必须参数化版本化入 config 哈希（否则违反协议 §8.1 可比性）。

---

## 6. 推荐结论与致命风险

### 6.1 推荐结论

1. **做不做 DIY：做，但只做"Gate1 前协议预研 pilot"，不做正式验证设备。**
2. **做哪条（按优先级）**：
   - **P0（推荐）**：**二手 Tobii 4C / EyeTribe 存量复用**（若 ¥400–1100 二手可得）——精度 0.5–1° 是 DIY 家族上限、数据流最贴近既有范式，可跑通"采集→重建→指标→bootstrap"全链 + 标定/质控流程，并给出一组**粗 AOI 级**参照数字。
   - **P1（零成本兜底）**：**L2CS-Net 纯软件 webcam**（MIT、现成 `l2cs` 包）——零硬件成本、最快跑通协议全链，但精度约 3–10°，仅用于**全链联调 + 粗 AOI 演示**，数字不可用于一致性结论。
   - **不推荐**：EyeTrackVR（VR 场景不符 + 非商业 copyleft 许可）、手机前置（隐私/精度/导出三重风险）、树莓派自建屏幕式（回报/工时比最低）。
3. **定位（组合策略）**：DIY pilot 的价值 = **提前验证协议实施细节**（设备适配器接口、标定与漂移校正流程、注视聚类参数、有效标志规则、letterbox 逆变换在自有分辨率下的正确性、bootstrap/表 A–E 管线复用），使商业设备（Fusion 250，降级 ET5）到货后能"零返工复测"。**DIY 数据与商业设备数据严格分表、不混表**（协议 §8.1）；DIY 一致性数字一律标注"预研设备，非正式验证"。
4. **正式一致性数字**：仍以 **A2 首选 Fusion 250 / 降级 ET5** 为准（与 A2 结论无冲突）。DIY 不替代、不延迟商业设备采购 Gate1。

### 6.2 致命风险（DIY 路线）

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| **精度不足导致一致性数字不可解释** | σ_e≥1–10° 时 NSS/IG/sAUC 退化为粗 AOI，数字无意义甚至误导 | 严格按 §4.2 阈值分层：仅粗 AOI 用途；σ_e 实测（标定验收 + 漂移校正）并写入 limitations |
| **标定失败率高** | 自建/二手设备无研究级校准，眼镜/低照度/头动致高排除率，样本报废 | 提前小范围试采；建立"标定验收精度→排除→记录"流程（对齐 A2 §3.4） |
| **社区项目停维护** | PyTribe 2016 停更、Tobii 4C 开源驱动 2020 停更、L2CS 2024 停更 → 依赖腐烂、无人修 bug | 只把 DIY 当一次性预研工具，不纳入长期管线；锁定版本 + 冻结依赖 |
| **许可（GPL/非商业 copyleft）传染** | PyTribe/TobiiGlassesPyController 为 GPL-3.0；EyeTrackVR 为非商业 copyleft 自定义许可 | 内部工具、**不对外分发、不入公开仓库**则 GPL 不触发分发义务（内部使用不在 GPL 分发范围内）[推断，法务确认前不对外]；EyeTrackVR 因许可更严且场景不符，直接不采用 |
| **二手硬件无质保/停产** | 设备损坏无法替换，实验中断 | 二手渠道验证后再投入；备 L2CS 纯软件兜底 |
| **隐私/保密红线** | 部分手机 app 上传视频违反"禁外发"红线 | 仅选本地-only 采集；手机路线已不推荐 |

---

## 7. 来源清单

### 7.1 外部 web 来源（访问日期均为 2026-09-13）

| # | 来源 URL | 取得的关键事实 | 状态 |
| --- | --- | --- | --- |
| 1 | https://api.github.com/repos/Ahmednull/L2CS-Net | L2CS-Net：MIT、527★/114 fork、Python/PyTorch、最近推送 2024-02-02、Gaze360/MPIIFaceGaze 权重 | 已核实 |
| 2 | https://raw.githubusercontent.com/Ahmednull/L2CS-Net/master/README.md | `l2cs.Pipeline` 用法、ResNet50、Gaze360/MPIIFaceGaze 训练、模型 pkl 下载 | 已核实 |
| 3 | https://api.github.com/repos/EyeTrackVR/EyeTrackVR | 1143★/93 fork、最近推送 2026-09-07、license=other（NOASSERTION）、topics 为 VR 社交（chilloutvr/vrchat） | 已核实 |
| 4 | https://raw.githubusercontent.com/EyeTrackVR/EyeTrackVR/main/LICENSE | **Babble Software Distribution License 1.0（2025-01）：非商业 + copyleft + 禁商用硬件集成** | 已核实 |
| 5 | https://api.github.com/repos/pupil-labs/pupil | pupil 主仓 **LGPL-3.0**、1732★/705 fork、最近推送 2026-08-31、Python | 已核实 |
| 6 | https://api.github.com/repos/esdalmaijer/PyTribe | PyTribe：GPL-3.0、28★、最近推送 2016-07-22（停维护） | 已核实 |
| 7 | https://ait.ethz.ch/xgaze | ETH-XGaze：1M+ 样本、110 参与者、**CC BY-NC-SA 4.0**、ECCV 2020、代码 xucong-zhang/ETH-XGaze | 已核实 |
| 8 | https://api.github.com/search/repositories?q=tobii+4c+eye+tracker&sort=stars&order=desc&per_page=8 | Tobii 4C 开源生态：`Eitol/tobii_eye_tracker_linux_installer`（177★/C/2020 停更/无许可）、`GazeOSC`（MIT/2022 停更）、`DigitalNatureGroup/TobiiEyeTracker.py`（2022 停更）、历史"4C 取注视需授权" | 已核实 |
| 9 | https://api.github.com/search/repositories?q=tobii+glasses&sort=stars&order=desc&per_page=5 | tobii-glasses 指 Tobii Pro Glasses 2/3（穿戴式）：`ddetommaso/TobiiGlassesPyController`（GPL-3.0/62★/2024-08）、`tobiipro/Tobii.Glasses3.SDK`（已归档） | 已核实 |
| 10 | https://www.opengaze.io/ 与 https://github.com/opengaze/opengaze | OpenGaze 开源眼动项目存在（屏幕式 DIY 参照，未取到 LICENSE/活跃度细节） | 已核实存在（细节推断） |
| 11 | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker | MediaPipe Face Landmarker 官方文档入口（本环境页面跨域重定向未取到原文，iris landmark 与精度为推断） | 未核实（重定向，正文标注推断） |
| 12 | http://gaze360.csail.mit.edu/ | Gaze360 项目页（**403 Forbidden**，未取到原文；数据集信息转引自 L2CS README #2） | 未核实（403） |

### 7.2 项目内既有文档引用（非外部来源，仅作口径引用）

- [docs/research/benchmark-protocol.md](../research/benchmark-protocol.md)（§2.1/§2.4/§5/§6.0/§6.9/§7.2/§8.1/§8.4）
- [docs/proposals/eyetracker-validation-proposal.md](eyetracker-validation-proposal.md)（A2：§2.1 对比表/§2.2 许可注记/§2.3 推荐配置/§3.4 校准/§4.1 适配器需求）
- [src/ui_attention/metrics/eval/ueyes_driver.py](../../src/ui_attention/metrics/eval/ueyes_driver.py)、[groundtruth.py](../../src/ui_attention/metrics/eval/groundtruth.py)（日志解析/letterbox 逆变换/`FixationSet` 契约）

---

## 8. 证据分级汇总

- **已核实（2026-09-13 一手抓取）**：L2CS-Net MIT 许可与停更状态（API+README）；EyeTrackVR 活跃度、VR 定位、自定义非商业 copyleft 许可全文；pupil 主仓 LGPL-3.0 许可与活跃度；PyTribe GPL-3.0 与 2016 停更；ETH-XGaze CC BY-NC-SA 4.0 非商用与体量；Tobii 4C 开源生态碎片化与停更（多仓库 API）；tobii-glasses 指穿戴式 Pro Glasses 2/3（非屏幕 4C）。
- **推断**：各 DIY 路线精度量级（EyeTribe≈1°、4C≈0.5–1°、头戴/屏幕式 IR≈1–3°、L2CS≈3–10°、MediaPipe≈3–5°）；§4.2 的 σ_e→σ_eff 噪声阈值（协议 σ=1° 口径下的几何/信号处理推导，非单一文献阈值）；BOM 与工时（区间估计）；许可"内部使用不触发 GPL 分发义务"（法务结论除外）。
- **未核实**：MediaPipe Face Landmarker 原文（页面重定向）；Gaze360 项目页（403）；L2CS-Net 论文精确 MAE（PDF 未取）；EyeTribe/4C 二手实际成交价（需实购验证）；各研究级/消费级设备精确精度（需规格书）。

*本文档为 A3 子任务独占产出（单文件）；不修改任何其他文件、不下载二进制/数据集/硬件设计文件、不写 Git。*

---

## 9. L2 裁决记录（2026-09-13，二级总控）

1. **σ_e 噪声阈值（§4.2）冻结为"工作解释阈值 v1"**：≤0.5° 可支撑 NSS/IG/sAUC；0.5–1° 边际（仅方向性/粗验证，limitations 必写）；≥2° 仅粗 AOI。定性维持 A3 标注：系协议 σ=1° 口径下的几何/信号处理推导，**非文献阈值**；Gate2 须以实测校准验收 σ_e 逐设备复核本阈值，DIY 与商业设备同屏对照数据到手后做一次经验校验并递增版本。
2. **适配器工程要求冻结**：设备适配器实现时，注视聚类（I-VT/I-DT 速度阈值/最小时长）、有效标志等价规则、标定映射层参数（视线角/iris→屏幕交点）**必须版本化入 config_hash**（协议 §8.1 可比性），否则该运行产物不得与既有表并列引用。
3. **许可工作假设**：内部使用不触发 GPL 分发义务维持为工作假设（推断），任何对外分发/开源发布前须法务确认；EyeTrackVR 非商业 copyleft 许可不采用之结论维持。
4. **DIY 定位确认**：采纳 A3 推荐——DIY 仅作 Gate1 前协议预研 pilot（P0 二手 4C/EyeTribe、P1 L2CS-Net 兜底），不替代不延迟商业设备采购；DIY 数据与商业数据严格分表；二手可得性与成交价属"需实购验证"，是否启动 pilot 采购由用户/组织批示（与 Gate1 独立的小额决策）。
