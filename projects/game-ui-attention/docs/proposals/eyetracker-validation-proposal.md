# 眼动真验证立项建议书（A2）

- 文档日期：2026-09-13
- 起草子任务：A2（眼动真验证立项建议书线），向二级总控（L2）汇报
- 状态：立项建议书（供组织决策，尚未获采购/执行批准）
- 性质：**可公开脱敏文档**——不含内部截图、凭据、个人身份信息、本机绝对路径；价格一律标注为调研区间并注明"报价需询价"；SGAME 界面仅以界面族类别（结算/商店/任务/弹窗）指代，不含具体画面内容。
- 证据分级：`已核实` = 本任务 2026-09-13 直接抓取一手页面原文；`推断` = 由证据合理推导但未实测；`未核实` = 未取得一手来源（附原因）。设备精度/价格为厂家标称或行业通行口径，一律以正式规格书/报价单为准。

---

## 0. 摘要与建议速览（结论先行）

1. **立项必要性与边界**：主后端 `foveacast-onnx-3s-v1` 的训练分布为西方桌面/移动 UI，**不含游戏 UI**，其在 SGAME 界面族上的预测有效性至今为"未验证"（见 [benchmark-protocol.md §0.2](../research/benchmark-protocol.md)、[validation-plan.md §4.2](../validation-plan.md)、[g2-comparison.md 未验证项](../acceptance/g2-comparison.md)）。本建议书主张：**用自采真实眼动数据验证模型与人类注视的一致性**，首轮目标 = 一致性数字（NSS/IG/sAUC），**不是微调、不是发布**。
2. **设备首选**：**Tobii Pro Fusion 250（屏幕式研究级，250 Hz）**——与"受试者看屏幕截图"任务形态最匹配、官方 Python SDK 成熟、校准流程标准化。**降级**：Tobii Eye Tracker 5（消费级 135 Hz，低成本）；若未来转向"真实游戏内操作"而非看截图，改选 Pupil Labs Neon（穿戴式、免校准）。
3. **样本量**：界面 20–40 张 SGAME 截图（覆盖结算/商店/任务/弹窗等族），参与者 12–20 人（首轮可 12 人起，用图像级 bootstrap 报告不确定性，沿用 R2 §7.2 口径）。
4. **口径复用**：NSS/IG/sAUC 与中心偏置基线（CB）、S0–S2 判定、注视重建（BPOGV=1/时长窗/letterbox 逆变换）全部沿用 [benchmark-protocol.md](../research/benchmark-protocol.md)；自有设备日志只需新增一个"设备适配器"，指标/基线/bootstrap 层全部复用既有代码。
5. **许可红线**：数据与模型仅内部使用；商用前仍需关闭 G3（SALICON）/G4（VGG16）/G8（ImageNet）缺口（引用 [sources-and-decisions.md §3](../sources-and-decisions.md)）。
6. **决策门**：Gate1 采购批准 → Gate2 首轮数据质量 → Gate3 微调立项（仅当首轮一致性数字揭示模型不足时才进入）。

---

## 1. 目标与范围

### 1.1 目标

- **第一阶（本建议书主目标）**：验证 `foveacast-onnx-3s-v1` 在 **SGAME 界面族**上的预测有效性——即模型输出显著图与真实人类注视之间的一致性（NSS/IG/sAUC 口径沿用 R2 协议，见 [benchmark-protocol.md §6](../research/benchmark-protocol.md)）。
- **第二阶（可选，另行立项）**：若第一阶数字揭示模型在游戏 UI 上不足，用自有眼动数据对 MSI-Net/foveacast-training 训练栈做微调（微调路径许可盘点见 §4.2）。

### 1.2 范围边界（红线）

- **验证 ≠ 发布**：本立项产出的是一致性证据（模型 vs 人类注视），不是对外的"预测准确率"宣传，也不承诺"点击率/任务效率提升"（与 [validation-plan.md §5](../validation-plan.md) 可声明结论表一致）。
- **数据与模型仅内部使用**：SGAME 截图与自采眼动数据仅限内网本地存储与计算，**不上传任何云服务、不入公开 Git 仓库**（落位 `local-data/`，已 gitignore，见 [runtime-feasibility.md §2.2](../research/runtime-feasibility.md)）。
- **不含**：在线 A/B 产品指标、点击/操作预测、时序（视频/scanpath）模型、任务搜索模型（SeekUI 线单独存档，D013）。
- **结果表述红线**（沿用 [benchmark-protocol.md §8.4](../research/benchmark-protocol.md)）：不得把 NSS/IG/sAUC 数值称为"预测准确率"；不得由公开 UI 结果外推为游戏结论；不得省略窗口/基线版本而单独引用单一指标。

### 1.3 与既有口径的一致性声明

本建议书**只引用、不重述**既有口径，冲突处显式标注。关键依赖：

| 既有口径 | 出处 | 本建议书用法 |
| --- | --- | --- |
| 主后端 = foveacast-onnx-3s-v1（S2 窗口匹配口径通过硬门槛） | [g2-comparison.md](../acceptance/g2-comparison.md)、D011 | 验证对象 |
| NSS/IG/sAUC/CC/KL 精确定义、float64/log2、σ=1° 模糊 | [benchmark-protocol.md §6](../research/benchmark-protocol.md) | 指标口径直接沿用 |
| S0（自检）/S1（基线有效性）/S2（候选及格线）判定 | [benchmark-protocol.md §8.3](../research/benchmark-protocol.md) | 一致性评估判定口径 |
| 图像级 bootstrap（B=2000、种子 20260911、按类别分层）+ 配对差值 + Holm | [benchmark-protocol.md §7.2](../research/benchmark-protocol.md) | 不确定性报告口径 |
| 中心偏置基线 CB = 本域 train-only 构造、版本化哈希 | [benchmark-protocol.md §4.2](../research/benchmark-protocol.md) | **按 SGAME 域重建**（不沿用 UEyes 的 CB） |
| 注视重建 = BPOGV=1、TIME≤窗口前缀截断、1920×1200 letterbox 逆变换、越界丢弃 | [benchmark-protocol.md §2.1/§6.0](../research/benchmark-protocol.md)、[ueyes_driver.py](../../src/ui_attention/metrics/eval/ueyes_driver.py) | 自有设备适配器须对齐 |
| 游戏 UI 有效性维持"未验证"（无自采数据） | [benchmark-protocol.md §0.2](../research/benchmark-protocol.md) | 本立项拟关闭该缺口的验证侧 |
| 本机 RTX 4070 12GB / 64GB RAM / 无系统级 CUDA Toolkit / venv py3.12 | [runtime-feasibility.md §1](../research/runtime-feasibility.md) | 微调 GPU 可行性引用（§5） |
| 许可缺口 G3/G4/G8 商用前须关闭 | [sources-and-decisions.md §3](../sources-and-decisions.md)、[model-candidates.md §6.2](../research/model-candidates.md) | 微调/商用路径红线（§4.2） |

---

## 2. 设备选型对比

> 访问日期均为 **2026-09-13**。精度（°）为厂家标称或行业通行口径，正式采购前以**官方规格书/报价单**为准；价格仅为调研区间，**报价需询价**（研究级设备价格随配置、软件授权、保修、地区与促销浮动）。

### 2.1 对比表（6 方案）

| 维度 | Tobii Pro Fusion 250（屏幕式·研究级） | Tobii Eye Tracker 5（屏幕式·消费级） | SR Research EyeLink 1000 Plus（屏幕式·科研金标准） | Pupil Labs Neon（穿戴式·免校准） | Pupil Labs Core（穿戴式·开源） | WebGazer.js（纯摄像头·开源） |
| --- | --- | --- | --- | --- | --- | --- |
| 采样率 | **250 Hz**（另有 Fusion 120 = 120 Hz）[已核实] | **135 Hz** [已核实] | **2000 Hz（单眼）/1000 Hz（双眼）** [通行口径，需规格书] | **200 Hz（双眼眼相机）** [已核实] | **200 Hz 或 120 Hz（眼相机）** [已核实] | 摄像头帧率（典型 30–60 Hz，非专用）[推断] |
| 精度（°） | 约 **0.3°**（标称，需规格书）[未核实精确值] | **未正式公布**（推断 0.4°–0.9° 量级）[推断] | **0.25°–0.5°**（典型 0.15°，需规格书）[通行口径] | 约 **1.5°**（场景相机参考系，白皮书口径）[未核实精确值] | 约 **0.6°**（标称，需规格书）[未核实精确值] | **1°–5°**（大方差，头部移动敏感）[论文口径] |
| 校准复杂度 | 屏幕式点校准（5/9 点，SDK 提供 `ScreenBasedCalibration`）[已核实] | 自带自动/简易校准（消费体验，研究级校准能力受限）[推断] | 高精度校准（Hv9 等，科研标配）[通行口径] | **免校准**（端到端深度学习注视线估计）[已核实] | 需 3D 校准 + 相机内参估计（较复杂）[已核实] | 自校准（点击/光标交互）[已核实] |
| SDK/Python 支持 | **Tobii Pro SDK，Python（`tobii_research` PyPI，找设备/校准/订阅注视）成熟** [已核实] | Tobii Stream Engine（消费 SDK，研究功能受限）[推断] | pylink（SR Research 官方 Python）[通行口径] | Real-time API + Recording API + 开源库（Python）[已核实] | 开源 Pupil Capture/Player/Service + 网络 API（Python）[已核实] | JavaScript（浏览器内），无原生 Python [已核实] |
| 价格区间 | 研究级设备**数万元人民币级**（报价需询价） | 约 **$195 USD**（≈¥1400 量级，报价需询价）[已核实为厂商建议零售价量级] | **数十万元人民币级**（报价需询价） | **数万元人民币级**（模块+伴侣手机，报价需询价） | **数万元人民币级**（报价需询价） | **免费**（开源，GPLv3/LGPLv3 见 §2.2）[已核实许可] |
| 内嵌工作室适配性 | **最适合**：屏幕式，直接对显示器上截图，与"看截图"任务、UEyes 原 Gazepoint 归一化坐标范式可对齐 | 可用作低成本降级：体积小、易安装，但 SDK/校准研究属性弱 | 精度最高但**过配**：贵、重、占桌面、需专门安装与培训 | 穿戴式适合真实操作/走动场景；看截图需 Reference Image Mapper 映射到屏幕，多一步 | 开源可定制，但校准复杂、维护成本高 | **仅粗粒度**：精度不足以支撑 NSS/IG 级一致性验证 |

### 2.2 关键许可/维护注记

- **WebGazer.js**：仓库已迁移至 `brownhci/WebGazer`（原 `mturk/webgazer` 404）；**GPLv3**，估值 <$100 万的公司可选 LGPLv3；**官方维护已于 2026-02-24 结束**（社区继续经 Issue 支持）[已核实 README]。其"客户端本地运行、不上传视频到服务器"对本项目的保密红线反而是加分项，但精度不达标。
- **Pupil Labs Neon 的红线提示**：Neon 的默认分析链路是 **Pupil Cloud（云端）**——本项目**禁用云上传**，必须走 **Neon Player 离线导出 + USB 传输**（见 §6.3）。
- **Tobii Pro SDK**：`tobii_research` 经 PyPI 安装，本机 venv py3.12 可承载（推理/采集主机复用 [runtime-feasibility.md §2](../research/runtime-feasibility.md) 的隔离环境原则）。

### 2.3 推荐配置（首选 + 降级）

| 档位 | 设备 | 理由 | 前提 |
| --- | --- | --- | --- |
| **首选** | **Tobii Pro Fusion 250** | 屏幕式研究级，采样率/精度/成熟 Python SDK 三满足；校准流程标准化；与"看截图自由观看/任务导向"任务形态天然一致；本域注视重建的"屏幕归一化坐标→letterbox 逆变换"可直接沿用协议口径 | 采购预算与审批（Gate1）；正式规格书确认精度 0.3° |
| **降级 1** | **Tobii Eye Tracker 5** | 低成本（消费级）、135 Hz 对本需求够用、安装零门槛 | 接受研究级 SDK/校准能力受限、精度未正式公布 → 一致性指标置信度下降，结论只作"粗验证" |
| **降级 2** | **Pupil Labs Neon** | 免校准、200 Hz、穿戴式 | 仅当研究目标转向"真实游戏内操作"而非看截图；屏幕映射需离线 Reference Image Mapper，**禁用 Pupil Cloud** |
| 不推荐为正式验证 | WebGazer.js | 精度 1°–5° 不足以支撑 NSS/IG/sAUC | 仅作方法探索/预研粗校验，不作为正式验证设备（§6.4） |
| 不推荐（过配） | EyeLink 1000 Plus | 精度最高但成本/占空间/培训负担远超本需求 | 若组织已有设备则可用，否则不为本立项采购 |

---

## 3. 研究协议草案（可直接照做）

### 3.1 界面样本

- **数量**：20–40 张 SGAME 界面截图。
- **覆盖族**：结算、商店、任务、弹窗（含通用弹窗/确认框）、HUD/主界面等典型族；每族 ≥4 张，避免只选模型表现好的样例（与 [validation-plan.md §3](../validation-plan.md) 一致）。
- **脱敏与保密（红线）**：截图仅内部使用；展示在眼动主机显示器上**本地运行**，不导出、不上传、不入公开仓库；采集时对截图做最低限度标识（如族类+编号），不携带账号/昵称/充值等个人信息。落位 `local-data/`（gitignore）。

### 3.2 参与者

- **人数**：12–20 人（首轮可 12 人起）。参考：UEyes 公开 UI 上限为 62 人（[benchmark-protocol.md §2.1](../research/benchmark-protocol.md)），本域内部首轮采用小样本 + bootstrap 报告不确定性（沿用 R2 §7.2 图像级重采样口径），不追求达到 62 人。
- **纳入/排除**：正常或矫正视力（可戴眼镜/隐形）；排除重度散光、眼部疾病、无法稳定注视者（校准失败即排除并记录）。
- **不确定性声明**（沿用 [benchmark-protocol.md §7.2](../research/benchmark-protocol.md) 限制声明口径）：参与者间变异不单独建模，bootstrap CI 仅反映图像抽样变异；首轮小样本的参与者维度限制须写入产物 limitations。

### 3.3 任务设计（两类）

1. **自由观看**：每张截图自由观看 **3s / 5s**（无任务指令）。
2. **任务导向**：给受试者一个明确目标（如"找到结算入口的位置"），观察目标导向下的注视差异。
- 两类任务**分开分析**（与 [validation-plan.md §4.2](../validation-plan.md)"自由观看和带任务观看分开分析"一致）。
- **窗口口径注记**：R2 协议窗口为 1s/3s/7s，本域建议 3s/5s。3s 与 `foveacast-3s` 训练/标称窗口构成"窗口匹配"（与 S2 已修订口径一致）；**5s 无对应模型窗口**，协议冻结时须明确 5s 是"跨窗口敏感性记录"还是改用 7s 模型作最近邻——此项提交 L1/L2 冻结（§7 待裁决）。

### 3.4 校准与质量控制

- **校准流程**：每名受试者开始时执行标准点校准；正式试次前做**漂移校正（drift check）**。
- **重校准阈值**（建议默认，冻结时确认）：漂移校正注视偏移 > **0.5°**（或每 10 张图 / 每 5 分钟间隔）触发重新校准；校准失败（无法收敛到验收精度）者排除并记录。
- **注视重建参数（沿用协议口径，[benchmark-protocol.md §2.1/§6.0](../research/benchmark-protocol.md)）**：
  - 有效性标志（对齐 Gazepoint `BPOGV` 语义，自有设备用其等价有效标志）只取有效注视；
  - 观看窗口 = 同一 trial 注视序列的**时长前缀截断**（3s 取 TIME≤3，5s 取 TIME≤5）；
  - 坐标映射 = 屏幕归一化坐标 → 显示分辨率像素 → **letterbox 逆变换**（截图按宽高比居中放置，越界注视丢弃并计数入审计）。
- **图像有效性门槛（沿用协议 §3.5）**：窗口内有效注视点 < 10 或观看者 < 3 的（界面×窗口）组合标记 `low_fixation`/`low_viewers` 并默认从该窗口指标排除。

### 3.5 知情同意与隐私

- **知情同意书**：说明目的（模型一致性验证）、时长、可中途退出、数据用途（仅内部研究）。
- **数据本地存储**：原始日志与聚合真值仅存内网本机（`local-data/`），不传云、不对外。
- **匿名化**：参与者以编号标识（如 P01–P20），不关联姓名/工号；报告只呈现聚合统计。
- **删除权**：参与者有权要求删除其原始数据；删除后重跑指标须记录并递增协议版本号。

---

## 4. 与现有管线的衔接

### 4.1 一致性评估（模型 vs 人类注视）

既有代码 [metrics/eval/](../src/ui_attention/metrics/eval/) 已实现 UEyes 评估全链路，本立项**最大限度复用**：

| 环节 | 既有实现 | 本立项处理 |
| --- | --- | --- |
| 日志流式解析 | `ueyes_driver.iter_log_rows`：BPOGV==1、TIME≤窗口、列名嗅探（含 `TIME(...)` 后缀） | **新增设备适配器**（自有眼动仪 CSV 列名/坐标系统 ≠ Gazepoint），参数化列映射与坐标映射 |
| 坐标映射 | `ueyes_driver.screen_to_image_point`：1920×1200 letterbox 逆变换 | 复用逆变换数学，显示分辨率参数改为实际采集分辨率 |
| 真值聚合 | `ueyes_driver.build_groundtruth` / `groundtruth.FixationSet` / `blurred_truth`（σ=1° 模糊、sum=1） | **直接复用**（跨参与者按图聚合、计数/时长加权） |
| 指标计算 | `metrics.py`：IG/NSS/CC/sAUC/AUC-Judd/KL | **直接复用**（NSS/IG/sAUC 为主，CC/KL/AUC 佐证） |
| 中心偏置基线 | `baselines.CenterBiasBaseline`（train-only、64×64、σ_bin=1） | **按 SGAME 域重建**（不沿用 UEyes 的 CB） |
| 不确定性 | `bootstrap`：图像级 B=2000、配对差值 + Holm | **直接复用**（本域类别层=界面族，替换原 webpage/desktop 等类别标签） |
| 判定 | `pipeline` S0–S2 | **直接复用**（S2 沿用窗口匹配口径） |

**需新增的工程件**（不在本立项建议书范围外改动共享代码，仅列需求）：① 设备适配器（日志→`FixationSet`）；② 本域划分文件（无官方 train/test，用协议 §3.3 确定性哈希 + §3.4 反泄漏五规则自产并冻结）；③ 本域 CB 构造（train-only）；④ 界面族类别标签。指标/基线/bootstrap/判定层零改动。

### 4.2 微调路径（第二阶，可选）——训练栈许可盘点

| 资产 | 许可 | 本立项状态 | 出处 |
| --- | --- | --- | --- |
| foveacast-training 代码 | **MIT**（LICENSE 全文已验证） | 可内部微调 | [model-candidates.md §C2](../research/model-candidates.md) |
| MSI-Net 上游代码（Kroner） | **MIT**（已验证） | 可内部微调 | 同上 |
| UEyes 预训练权重（foveacast ONNX 的微调起点） | MIT 代码 + **CC-BY-4.0 署名传递**（Jiang et al. 2023） | 下游携带署名即可 | 同上 |
| **自有数据（SGAME 眼动）** | 自采，**内部使用不触发对外条款** | 无对外义务 | 本建议书 §3.5 |
| **商用前仍须关闭** | **G3（SALICON 条款未核实）/G4（VGG16 原始权重条款）/G8（ImageNet 残留风险）** | **本次验证不涉商用，不受阻** | [sources-and-decisions.md §3](../sources-and-decisions.md)、[model-candidates.md §6.2](../research/model-candidates.md) |

**结论**：内部验证 + 内部微调在许可层面**无阻碍**；一旦走向对外分发/商用，须先关闭 G3/G4/G8 缺口（维持既有裁决口径，不在本建议书内重复裁决）。

### 4.3 数据契约落位

- 原始日志、截图、真值 npz、CB 文件、评估产物全部落位 `local-data/`（gitignore，见 [runtime-feasibility.md §2.2](../research/runtime-feasibility.md)）；划分/协议版本清单（JSON）可入 Git（不含数据本体，与 [benchmark-protocol.md §3.3](../research/benchmark-protocol.md) 一致）。
- 产物标注 `模型预测` 与"游戏 UI 一致性验证（自采数据，非公开数据）"字样（与 [validation-plan.md §6](../validation-plan.md) 模板一致）。

---

## 5. 成本 / 周期 / 人力估算与决策门

> 以下均为**区间估计 + 假设**，正式立项前由采购/财务核准。

### 5.1 成本估算表

| 项 | 区间 | 假设 / 依据 |
| --- | --- | --- |
| 设备采购（首选 Fusion 250） | 数万元人民币级（**报价需询价**） | 研究级设备随配置/软件/保修浮动；Fusion 120 更低（采样率降至 120 Hz） |
| 设备采购（降级 Eye Tracker 5） | 约 $195 USD（≈¥1400 量级，报价需询价） | 厂商建议零售价量级，2026-09-13 |
| 设备采购（穿戴式 Neon/Core） | 数万元人民币级（报价需询价） | 仅降级 2 路线 |
| 参与者激励 | 12–20 人 × ¥100–300/人 ≈ **¥1,200–6,000** | 内部招募，时长约 0.5–1 小时/人 |
| 研究执行（人天） | **10–15 人天** | 招募/排期、知情同意、设备搭建、校准与采集、数据落盘与质控 |
| 分析（人天） | **5–8 人天** | 设备适配器、真值重建、CB 构造、逐图指标、bootstrap、报告 |
| 微调实验（人天，第二阶） | **5–10 人天** | 数据准备（划分/增广）、超参、训练、导出 ONNX、重评估 |
| 总周期（第一阶，自 Gate1 通过起） | **6–10 周** | 含设备到货（数周）+ 招募 + 采集 + 分析 |
| 总周期（含第二阶微调） | **另加 3–5 周** | 仅当 Gate3 立项 |

### 5.2 微调 GPU 可行性（本机 RTX 4070 12GB）

- **引用 R3 实测结论**：本机 RTX 4070（12 GB 显存）+ 64 GB RAM + 无系统级 CUDA Toolkit；**所有候选模型训练显存均未实测**（[runtime-feasibility.md §5-U4](../research/runtime-feasibility.md) 标注"未验证"）。作者记录的 foveacast 全量微调在 Apple M4 MPS 上约 3.5 小时/30 epochs（作者报告，未复现，[model-candidates.md §C2](../research/model-candidates.md)）。
- **本建议书判断**：MSI-Net 为轻量架构（240×320 输入、VGG16 encoder 初始化、参数量小），12 GB 显存**按规模推断足够**（batch 8–16 级）——**标注"待验证"**；微调前须先以最小 batch 实测峰值显存（沿用 R3"doctor 实测空闲显存"原则，桌面常驻已占 ≈4.8 GB）。若不足，降 batch 或梯度累积，无需采购新 GPU。
- 结论：**GPU 不作为本立项的采购项**，标注待验证。

### 5.3 决策门

| 门 | 触发条件 | 通过判据 | 失败动作 |
| --- | --- | --- | --- |
| **Gate1 采购批准** | 本建议书经 L2 转呈 L1 批准 | 设备选型 + 预算 + 知情同意/伦理口径获批 | 不采购；走"不立项"现状用法（§6.5） |
| **Gate2 首轮数据质量** | 设备到货、首轮 12 人采集完成 | 校准达标率、有效注视覆盖率、注视重建可跑通（S0 自检过） | 重校准/重采；仍不达标则评估降级设备或终止验证 |
| **Gate3 微调立项** | 第一阶一致性数字出炉 | 若模型显著优于本域中心偏置（S2 判定）→ **无需微调**；否则立项微调 | 维持"验证结论"并记录模型在本域不足的证据 |

---

## 6. 风险与替代

| 风险 | 影响 | 缓解 / 替代 |
| --- | --- | --- |
| 招募困难（内部 12–20 人难凑） | 样本不足、不确定性大 | 12 人起步 + bootstrap 报告不确定性；延长周期或分批；绝不伪造注视 |
| 硬件审批/到货周期 | 总周期拉长 | 提前启动 Gate1；降级用 Eye Tracker 5 先跑预研，Fusion 到货后复测 |
| **游戏画面保密（红线）** | 泄露 SGAME 素材 | 截图仅内部本地展示；**禁外发云服务**——Pupil Neon 禁用 Pupil Cloud、改 Neon Player 离线 + USB；WebGazer 客户端本地（不动用其服务器）；数据不入 Git |
| 摄像头方案精度不足 | 若仅摄像头可用，NSS/IG 级验证不可行 | 降级目标为"粗粒度 AOI 一致性"（WebGazer 1°–5° 只能判大区），结论标注方法局限；不作为正式验证 |
| 设备 SDK/校准研究属性弱（消费级降级） | 一致性置信度下降 | 明确结论只作"粗验证"，并在报告限制声明中如实记录 |
| 5s 窗口与模型窗口不匹配 | 口径风险 | 协议冻结时明确 5s 的窗口语义（§3.3 注记），不静默混表 |
| **不立项情形** | 游戏 UI 有效性持续"未验证" | 现状用法：**比较模式（A/B compare）+ A1 线在建的内部粗标注校准工具**（人工标注注视作为弱真值，非眼动真值），游戏 UI 预测有效性维持"未验证"并如实对外表述 |

---

## 7. 结论建议段

1. **推荐路线**：设备首选 **Tobii Pro Fusion 250**（屏幕式研究级，250 Hz，官方 Python SDK），降级 **Tobii Eye Tracker 5**（消费级 135 Hz，低成本）；若研究目标从"看截图"转为"真实游戏内操作"，改选 **Pupil Labs Neon**（穿戴式免校准，禁用云）。
2. **样本量**：界面 **20–40 张**（覆盖结算/商店/任务/弹窗等族），参与者 **12–20 人**（首轮 12 人起，图像级 bootstrap 报告不确定性，沿用 R2 §7.2）。
3. **首轮目标**：产出**一致性数字**——`foveacast-onnx-3s-v1` 在 SGAME 界面族上的 NSS/IG/sAUC（vs 本域重建的中心偏置基线，S0–S2 判定口径），**不是微调、不是发布**。
4. **微调决策**：仅在首轮一致性数字揭示模型不足时经 Gate3 立项；本机 RTX 4070 12GB 按规模推断够用（待实测峰值显存验证），GPU 不作为采购项。
5. **待 L1/L2 冻结项（提请裁决）**：① 5s 窗口与模型窗口的对应口径；② 重校准阈值 0.5° 与采样间隔默认值；③ 排除门槛（注视 <10 / 观看者 <3）沿用值；④ 是否同时采集自由观看 + 任务导向两类，还是首轮只做自由观看。

---

## 8. 来源清单

### 8.1 外部 web 来源（访问日期均为 2026-09-13）

| # | 来源 URL | 取得的关键事实 | 状态 |
| --- | --- | --- | --- |
| 1 | https://www.tobii.com/products/eye-trackers/screen-based/tobii-pro-fusion | Fusion 250 Hz、双相机、USB/USB-C、≤24" 显示器、可三脚架、Tobii Pro Lab 配套 | 已核实 |
| 2 | https://developer.tobiipro.com/python/python-getting-started.html | `tobii_research`（PyPI）、`find_all_eyetrackers`、`ScreenBasedCalibration`、`EYETRACKER_GAZE_DATA` 订阅 | 已核实 |
| 3 | https://gaming.tobii.com/product/eye-tracker-5/ | Eye Tracker 5 = 135 Hz、消费级游戏设备 | 已核实（采样率）；价格量级经搜索补充，报价需询价 |
| 4 | https://pupil-labs.com/products/neon/ 与 https://docs.pupil-labs.com/neon/hardware/module-technical-overview/ | Neon 模块构成（双眼 IR 眼相机、场景相机、IMU、USB-C 接伴侣设备） | 已核实 |
| 5 | https://docs.pupil-labs.com/neon/data-collection/data-streams/ | 眼相机 200 Hz（192×192、硬件同步）、场景相机 30 Hz 1600×1200（103°×77°）、**免校准端到端 DL 注视**、自动注视/扫视（VOR 补偿）、IMU 110 Hz、Real-time/Recording API、Pupil Cloud 重算 | 已核实 |
| 6 | https://docs.pupil-labs.com/core/hardware/ | Core 穿戴式、眼相机 200 Hz 或 120 Hz、需校准与内参估计、开源软件 | 已核实 |
| 7 | https://www.sr-research.com/eyelink-1000-plus/ 与 https://www.sr-research.com/eyelink-portable-duo/ | EyeLink 1000 Plus / Portable Duo 产品存在（页面 JS 重，规格表截断） | 产品页已核实存在；2000 Hz/精度/pylink 为行业通行口径，需正式规格书 |
| 8 | https://github.com/brownhci/WebGazer 与 https://raw.githubusercontent.com/brownhci/WebGazer/master/README.md | WebGazer 浏览器内运行、自校准、**GPLv3（估值<$100万可 LGPLv3）**、官方维护 2026-02-24 结束、论文 Papoutsaki et al. 2016 | 已核实（README 原文） |

### 8.2 项目内既有文档引用（非外部来源，仅作口径引用）

- [docs/research/benchmark-protocol.md](../research/benchmark-protocol.md)（R2 协议：§2.1/§3.3/§3.4/§3.5/§4.2/§5/§6/§7.2/§8.3/§8.4）
- [docs/research/runtime-feasibility.md](../research/runtime-feasibility.md)（R3：§1/§2/§5-U4）
- [docs/research/model-candidates.md](../research/model-candidates.md)（R1：§C2/§6.2 缺口 G1–G8）
- [docs/acceptance/g2-comparison.md](../acceptance/g2-comparison.md)（G2 验收、S2 窗口匹配裁决）
- [docs/validation-plan.md](../validation-plan.md)（§3/§4.2/§5/§6）
- [docs/sources-and-decisions.md](../sources-and-decisions.md)（D011/D013、§3 缺口处置）
- [src/ui_attention/metrics/eval/ueyes_driver.py](../../src/ui_attention/metrics/eval/ueyes_driver.py) 与 [groundtruth.py](../../src/ui_attention/metrics/eval/groundtruth.py)（日志解析/真值重建/letterbox 逆变换）

---

## 9. 证据分级汇总

- **已核实**：各设备采样率（Fusion 250 / ET5 135 / Neon 200 / Core 200·120）；各 SDK/Python 支持形态（tobii_research、Pupil Labs API、WebGazer JS）；校准方式（Fusion 点校准、Neon 免校准、Core 需 3D 校准、WebGazer 自校准）；WebGazer 许可 GPLv3 与维护停止；Eye Tracker 5 建议零售价量级。
- **推断**：Eye Tracker 5 精度量级（未正式公布）；Core/Neon 精度量级（厂家标称转述）；RTX 4070 12GB 对 MSI-Net 微调的显存足够性（轻量架构推断，未实测）。
- **未核实**：Fusion/EyeLink/Neon/Core 的精确精度值（官方规格表未完整抓取或产品页 JS 重）；EyeLink 2000 Hz/0.25°–0.5°/pylink（行业通行口径，未取到官方规格页原文）；各研究级设备的确切报价（需询价）。

*本文档为 A2 子任务独占产出（单文件）；不修改任何其他文件、不下载二进制/数据集、不写 Git。*

---

## 10. L2 裁决记录（2026-09-13，二级总控冻结）

对 §7 四项提请的裁决（属新研究协议参数冻结，不改动 G2 已批准判据）：

1. **5s 窗口口径**：首轮以 **3s 自由观看为主窗口**（与 foveacast-onnx-3s-v1 标称窗口构成窗口匹配，S2 口径一致）；**5s 采集但仅作跨窗口敏感性描述记录**（无门槛判定、单独分表、不与 3s 混表）；**不采用 7s 模型作 5s 最近邻**（避免口径混杂）。
2. **重校准阈值与采样间隔**：冻结默认值=漂移校正偏移 >0.5° 或每 10 张图/每 5 分钟触发重校准；采样间隔=设备原生采样率（有效采样率入审计产物）。该组参数属 Gate2 可复审项：首轮数据质量不达标时可调整并递增协议版本号。
3. **排除门槛**：沿用冻结值（窗口内有效注视 <10 或观看者 <3 标记 low_fixation/low_viewers 并默认排除），与 R2 §3.5 及 config 冻结值一致。
4. **任务类型**：首轮**两类同采**，但首轮一致性数字（NSS/IG/sAUC vs 本域 CB、S0-S2 判定）**仅用自由观看**；任务导向为 5-8 屏 pilot 子集（协议验证+描述性对照），分开分析、不混入主表（与 validation-plan §4.2 一致）。

裁决效力：本记录为协议冻结基线；设备采购与执行仍须 Gate1（L1/组织批准）。
