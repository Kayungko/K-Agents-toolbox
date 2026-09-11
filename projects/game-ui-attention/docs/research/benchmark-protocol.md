# 公开眼动数据上的显著性模型评估协议（Benchmark Protocol）

版本：v0.1（研究产出，未实施）。撰写日期：2026-09-11。来源核查日期：2026-09-11（所有"已核实"条目均于当日在线核对）。

状态说明：本协议是评估脚本的实施口径，尚未运行任何评估；文中所有成功判据均为**建议，待一级总控确认**（见 §8.3）。协议引用的数据集与模型均**未下载、未实测**；体量与许可信息来自来源页面记录。

---

## 0. 边界与适用性声明（必读）

1. **本项目无自采眼动数据。** 本协议只覆盖在**公开眼动数据集**上评估显著性模型的统一口径，是[验证方案](../validation-plan.md)第 4 节"预测有效性"在公开数据侧的具体化。
2. **游戏 UI 上的预测有效性保持"未验证"。** 本协议使用的公开数据集均为通用 UI（网页、桌面软件、移动应用、海报），**不包含游戏 UI**。公开数据上的任何分数**不得外推为游戏 UI 结论**，也不得用于宣称"模型懂游戏界面"。
3. 本协议不覆盖：在线 A/B 产品指标、点击/操作预测、时序（视频）模型、注视顺序或首眼位置推断、任何"真实注视时长/玩家看到的概率"表述。
4. 模型是预测分布，不是真实眼动。任何单一指标数值不得被称为"预测准确率"（与[验证方案](../validation-plan.md)§4、§5 一致）。
5. 本文档可公开：不含会话标识、凭据、个人身份信息、本机路径或受限第三方内容。数据集许可与署名要求见 §2 与 §10。

---

## 1. 目的与范围

目标：给出"工程师能直接照着写评估脚本"的最小完整口径，使任意候选显著性后端（如 DeepGaze IIE 等，见[技术方案](../technical-design.md)§3）在公开 UI 眼动数据上的评估结果**可复现、可比较、可审计**。

范围：

- 输入：静态完整 UI 截图 + 显著性模型输出的概率热图；
- 真值：公开数据集的注视点（fixations）/官方聚合热图；
- 输出：逐图指标、聚合统计表、配置指纹与审计报告；
- 用途：后端选型（哪个候选模型在公开 UI 数据上达到可用水平）、基线合理性检查；**不用于**游戏 UI 有效性声明。

不在范围：训练或微调模型（第一版不从零训练，见 README）；游戏截图评估（无真值眼动数据）。

---

## 2. 数据集

### 2.1 主数据集：UEyes（CHI 2023）

**引用**：Yue Jiang, Luis A. Leiva, Hamed Rezazadegan Tavakoli, Paul R. B. Houssel, Julia Kylmälä, Antti Oulasvirta. "UEyes: Understanding Visual Saliency across User Interface Types." CHI '23, Article 285, 1–21. DOI: [10.1145/3544548.3581096](https://doi.org/10.1145/3544548.3581096)。

**来源（2026-09-11 已核实）**：

| 项 | 内容 | 来源 |
| --- | --- | --- |
| 代码仓库 | <https://github.com/YueJiang-nj/UEyes-CHI2023>（数据处理、评估、UMSI++/DeepGaze++/PathGAN++） | GitHub README（已核实） |
| 项目页 | <https://userinterfaces.aalto.fi/ueyeschi23>（论文 PDF、附录、模型权重 zip、数据集链接） | 项目页（已核实） |
| 数据托管 | Zenodo record 8010312，DOI: 10.5281/zenodo.8010312，2023-06-06 发布 v1 | <https://zenodo.org/records/8010312>（已核实） |
| 许可 | **CC-BY-4.0**（Zenodo 记录 Rights 栏；论文最终版同为 CC BY，见 Aalto 研究门户） | Zenodo 页 + <https://research.aalto.fi/en/publications/ueyes-understanding-visual-saliency-across-user-interface-types>（已核实） |
| 体量 | 单文件 `UEyes_dataset.zip` **12.9 GB**，md5 `c2d53e6af0a47e1f459416d6839ec2c1` | Zenodo 页（已核实）。**本阶段不下载**；下载后必须校验 md5 并记录 |
| 规模 | 62 名参与者 × 1,980 张 UI 截图；4 个类别：webpage、desktop UI、mobile UI、poster | GitHub README + Zenodo 摘要（已核实） |
| 观看窗口 | **1s / 3s / 7s** 三档（`fixmaps_{1s,3s,7s}`、`heatmaps_{1s,3s,7s}`、`paths_{1s,3s,7s}`） | 仓库 README 目录结构 + 数据处理代码（已核实，见 §5） |
| 眼动仪 | Gazepoint（原始日志列定义见 Gazepoint API v2.0 文档，README 给出链接） | 仓库 README（已核实） |

**数据组织（来自仓库 README 与官方代码，已核实）**：

```text
UEyes_dataset.zip
├── images/                     # 全部 UI 截图，按实验 block 分目录（"block N"）
├── eyetracker_logs/            # Gazepoint 原始日志 CSV：{block}_hk0{YY}_fixations.csv
│                               #   （YY=参与者编号，参与者目录编号见 kh000–kh065）
│                               #   关键列：MEDIA_NAME（图片名）、TIME（时间戳 s）、
│                               #   FPOGD（注视时长）、BPOGV（有效性标志）、
│                               #   BPOGX/BPOGY（归一化屏幕坐标 0–1）
├── saliency_maps/              # 官方聚合真值（跨参与者、按窗口）
│   ├── fixmaps_{1s,3s,7s}/     #   二值注视存在图（0/255 8-bit PNG）
│   ├── heatmaps_{1s,3s,7s}/    #   高斯模糊热图（max 归一化到 0–255 8-bit PNG）
│   └── overlay_heatmaps_*/     #   叠加展示图（不用于计算）
├── scanpaths/paths_{1s,3s,7s}/ # 每图每参与者一张扫描路径渲染图（不用于热图指标）
├── info.csv                    # 每图：图片名、类别、block 编号、**train/test 官方划分标志**
└── README.md
```

**官方真值生成管线（已从 `data_processing/generate_heatmaps.py`、`aggregate_data.py` 源码核实）**：

1. 屏幕假设 1920×1200，图像按宽高比 letterbox 居中放置；BPOGX/Y 屏幕归一化坐标 → 图像像素坐标（减 padding、乘缩放）；**落在图像边界外的注视被丢弃**。
2. 窗口截断：`get_coordinates(t)` 取 TIME ≤ t 的注视 → 1s/3s 窗口是**同一 trial 注视序列的前缀截断**，7s 为全程（聚合代码 `duration=7`）。
3. 聚合：按图像合并**全部看过该图的参与者**的注视；支持两种加权：计数（unweighted）或注视时长加权（weighted，README 示例 `-w 1`）。
4. heatmap = 注视计数/加权和 → 高斯模糊 `sigma = 40 / scalar`，`scalar = min(1920/w, 1200/h)`（即 **σ ≈ 40 px @1920×1200，官方注释明确对应约 1° 视角**）→ max 归一化到 0–255 → 8-bit PNG。
5. fixmap = 注视二值存在图 → 高斯模糊 `σ = min(w,h)/400` → 阈值 `>0.001` 二值化 → 0/255 PNG。
6. 已知细节："some users did not see all the images in the block"（部分参与者未看完 block 内全部图片）→ **每图观看者数不均匀**，实现时必须统计（见 §7.3 表 D）。

**已知问题（必须在报告中声明）**：

- **勘误**：项目页声明"We regenerated Table 2 in the paper after fixing some calibration error in the dataset"——论文表 2 的模型分数是在数据校准修正后重新生成的；引用论文分数时必须注明版本与口径（见 §8.2）。
- 官方聚合热图为 **8-bit PNG**（256 级量化 + max 归一化），对 CC/KL/IG 等分布型指标有精度损失；**本协议默认从 `eyetracker_logs` 原始注视点重建真值**（按 §6.0 口径），官方 PNG 仅作交叉校验。
- 官方评估代码（`evaluation/eval.py`、`eval_heatmaps.py`）的若干指标口径与 Kümmerer/MIT 标准**不一致**（IG 无真实基线、KL 方向相反、AUC 用 0.7 相对阈值等），逐项差异见 §6.8；**本项目复现分数不得与 UEyes 论文表格分数直接混表**。

### 2.2 对照数据集 1：FiWI（Fixations in Webpage Images，ECCV 2014）

**引用**：Chengyao Shen, Qi Zhao. "Webpage Saliency." ECCV 2014。

**参数（2026-09-11 经 MIT Saliency Benchmark 数据集清单页核实，<http://saliency.mit.edu/datasets.html>）**：

- 149 张网页截图，3 类（Text 50 / Pictorial 50 / Mixed 49）；分辨率 1360×768，1° 视角 ≈ 26 px；
- 11 名观察者（21–25 岁），自由观看 5 s，EyeLink 1000（1000 Hz）；
- 官方页：<http://www.ece.nus.edu.sg/stfpage/eleqiz/webpage_saliency.html>。

**许可**：**未验证**（官方页在本环境核查时网络不可达，代理持续故障）。使用前必须重新核查官方页的下载与使用条款，必要时联系作者确认；未确认许可前**只作方法对照，不分发、不入仓**。

**用途与限制**：样本小（149 图 × 11 人），只用于**跨数据集方向一致性检查**（候选模型在网页类 UI 上是否与 UEyes-webpage 类别同向），不进入成功判据的定量门槛。仅单一 5 s 自由观看条件，无多窗口。

### 2.3 对照数据集 2（候选，未采纳）：WebSaliency（WWW 2021）

网页 UI 显著性基准。**核查状态：未验证**——原 GitHub 仓库地址（`github.com/shenyi-z/WebSaliency`）2026-09-11 核查返回 404；学术搜索 API 当日限流，未能二次确认论文与数据的当前可获取性。**不纳入本协议评估集**；若后续复核可获取且许可允许，再按 §3–§7 同口径扩展。

### 2.4 方法学参照（非 UI、不评估）

- **CAT2000**（4000 图、每图 24 人、5 s 自由观看、train/test 划分含留出观察者，arXiv:1505.03581）、**MIT300**（300 图、39 人、3 s，测试注视**不公开**以防过拟合）、**MIT1003**（1003 图、15 人、3 s）：自然图像基准，参数经 MIT 基准清单页核实（2026-09-11）。仅用于：指标口径参照、中心偏置构造惯例参照、理解候选模型的预训练来源（多数通用显著性模型在 SALICON/MIT1003 类自然图像上预训练）。**不作为本项目的 UI 评估集**。
- MIT 基准站已于 2019-12 停止维护，继任者为 <https://saliency.tuebingen.ai>（Kümmerer 组；本环境核查当日网络不可达，标注未验证）。
- **σ≈1° 视角约定**：MIT 基准页明确"连续注视图由注视点与高斯卷积生成，σ 通常取约 1° 视角（fovea 尺寸估计，Le Meur & Baccino 2013）"；UEyes 官方管线同口径（σ=40px@1920×1200）。本协议采纳（§6.0）。

### 2.5 游戏 UI 眼动数据：缺失声明

截至 2026-09-11，在本环境可达的检索渠道内**未能核实到公开、可下载的游戏 UI 眼动数据集**。这是"游戏 UI 预测有效性未验证"（§0.2）的直接数据侧原因之一。若一级总控后续提供或批准获取游戏眼动数据，需另立协议增补，不得默认沿用本协议结论。

### 2.6 数据获取与记录规则

- 本阶段（协议定义）**不下载任何数据集**；体量、md5、许可仅记录来源页面。
- 正式评估前下载时：校验 md5 → 记录 `{数据集, 版本/发布日期, 体量, md5, 许可, 引用}` 到运行 manifest；数据与真值 PNG **不入 Git**（与[技术方案](../technical-design.md)§10 一致）。
- CC-BY-4.0（UEyes）要求署名：所有对外报告必须包含 §2.1 的完整引用与 DOI。

---

## 3. 划分规则（split）

### 3.1 划分单位与分组

- **划分单位 = 图像（界面）**。同一图像的所有参与者注视聚合为该图的真值；**绝不按参与者划分**——参与者是同一图像真值的重复观测，不是独立评估样本（同界面不同参与者的处理 = 聚合，见 §6.0）。
- 分组维度：UEyes 的 `info.csv` 提供 **类别（webpage/desktop/mobile/poster）与 block 编号**；类别用作分层与细分报告维度，block 用作来源聚簇的近似（同 block 图片在实验与素材来源上相邻）。
- FiWI 按其 3 个网页类别分层（若采纳）。

### 3.2 官方划分优先（UEyes）

- `info.csv` 含每图 **train/test 官方标志**（仓库 README 已核实字段含义）。**默认采用官方标志作为冻结划分**：train 划分用于估计中心偏置基线与任何数据驱动配置；**test 划分用于全部正式指标报告**。
- 官方划分的产生准则（比例、是否按类别/block 分层）在论文正文中；本环境无法解析 PDF（web_fetch 不支持 PDF，ACM DL 反爬 403），标注**待补核**。实施时：下载后直接读取 `info.csv`，统计各类别/各 block 的 train/test 计数并写入划分清单（表 D），**不重新发明划分**。
- 若 `info.csv` 标志缺失、含义不明或与 README 描述冲突：触发 §3.3 备选划分，并在报告中记录原因。

### 3.3 备选自产划分（完全确定、可冻结）

仅当官方划分不可用时启用：

1. 取全部有效图像 ID，按 `(category, block, image_id)` 排序；
2. 对每图计算 `h = SHA256(protocol_version + "|" + image_id)`，按 `h mod 100` 分桶：桶 0–69 → train，70–84 → val（可选，仅用于超参/σ 敏感性），85–99 → test；
3. 生成 `splits.v1.json`：每图 `{image_id, category, block, split}` + 生成参数（protocol_version、算法、盐）+ 文件 SHA256；
4. **冻结**：清单入库（Git 跟踪的是清单 JSON，不是数据），此后只读；任何修改必须递增版本号（`splits.v2.json`）并**重跑全部对比**，新旧版本结果不得混表（§8.1）。

### 3.4 反泄漏规则

1. **近似重复截图不得跨划分**：对全部图像计算感知哈希（pHash 或 dHash，64 bit），任意两图 Hamming 距离 ≤ 8（建议默认，冻结时确认）判定为近似重复，归入同一重复簇；**每簇整体分到同一 split**。UEyes 覆盖网页/桌面/移动 UI，同一网站或应用的多状态截图（不同滚动位置、不同弹窗）预期存在，此规则强制它们同侧。
2. **同源分组**：若来源信息可考（同网站域名、同应用），按来源分组整体划分；UEyes 官方未提供逐图来源字段时，以 block + 感知哈希簇为近似，并在审计报告记录"来源级泄漏无法完全排除"为已知限制。
3. **真值构造隔离**：中心偏置基线（§4.2）、模糊 σ 敏感性、任何从注视数据估计的超参，**只能用 train 划分**（或 train 内交叉验证）；test 划分的注视数据不得参与任何基线/先验/后处理参数的构造。
4. **模型侧隔离**：候选模型若曾在 UEyes（或其子集）上训练/微调（如 UMSI++ 即用 UEyes 官方 train 划分训练，仓库已核实其训练脚本），则该模型在 train 划分上的分数**禁止报告为泛化性能**；正式表只报 test 划分。使用官方预训练权重（如 UMSI++ 权重）时，在表 E 配置指纹中记录"权重含 UEyes-train 信息"。
5. **泄漏审计清单**（每次评估运行必查）：划分文件哈希一致；近似重复簇无跨 split；基线文件生成时间早于 test 指标计算且仅引用 train 数据；被排除图像列表（§6.0 质量门槛）在 train/test 两侧分别记录。

### 3.5 图像有效性门槛（建议默认，冻结时确认）

- 窗口内有效注视点数 < 10 的 (图像, 窗口) 组合：标记 `low_fixation`，**默认从该窗口指标中排除**并计入审计表 D；
- 观看者数 < 3 的图像：标记 `low_viewers`，同上处理；
- 排除规则一经冻结不得中途更改；更改即新 protocol_version。

---

## 4. 基线（baseline）

基线是**版本化配置**（与[技术方案](../technical-design.md)§5 中心偏置条款一致）：文件 + 生成参数 + 哈希入库，报告表 E 记录版本。

### 4.1 均匀分布基线 U

- 定义：`U(p) = 1/N`，N 为评估网格像素数；每图相同。
- 用途：(a) 度量口径 sanity check——U 的 NSS、CC 期望≈0，AUC/sAUC 期望≈0.5，IG(以 U 为基线) 期望≈0；(b) 报告表中的"地板行"；(c) IG 的可选基线（`IG_uniform`）。

### 4.2 中心偏置基线 CB（数据驱动，非参数默认）

**构造方法 A（默认，非参数经验分布）**——从 **train 划分**注视点估计：

1. 对 train 划分内、指定窗口 w 的全部注视点，转换为图像归一化坐标 `u = (x + 0.5)/W, v = (y + 0.5)/H`（像素中心约定）；可选时长加权（与 §6.0 真值加权方案保持一致）；
2. 构建 2D 直方图 `H_cb`（bin 建议 64×64，冻结时确认），计数归一化为 sum=1；
3. 平滑：与 bin 尺度匹配的小 σ 高斯（建议 σ_bin = 1 bin）消除量化伪影；
4. 求值：对任意图像，将 `H_cb` 双线性上采样到该图评估网格，再归一化 sum=1；
5. 产物：`cb_{dataset}_{window}_{split}.v1.npz`（直方图 + 参数 + 源划分哈希 + 注视计数）。

**构造方法 B（敏感性分析用，参数径向模型）**：各向异性高斯 `CB(u,v) ∝ exp(-((u-μu)²/2σu² + (v-μv)²/2σv²))`，`(μu,μv,σu,σv)` 在 train 划分注视点上极大似然拟合；与 A 的差值作为"中心偏置形状敏感性"报告，**不替代 A**。

**规则**：

- CB **按数据集 × 窗口 × 划分分别估计**（1s/3s/7s 的中心偏置不同，注视随时间扩散）；
- CB 只能由 train 划分构造（§3.4.3）；
- CB 自身也作为"模型"跑完整指标管线，在报告表 A 中占一行（`model=center_bias`）；
- **不得**把 UEyes/FiWI 估计的 CB 宣称为"游戏玩家的中心偏置"（与技术方案 §5 一致）。

### 4.3 打乱注视基线 SF（可选参照，非默认）

- 定义：图像 i 的"预测" = **同划分内其他图像**的注视聚合图（模糊、归一化后）；即 empirical fixation prior across images。
- 用途：近似"人类跨图一致性"参照行，给 IG/CC 一个经验上界感（低于真值上界）；计算成本高（O(n²) 或抽样近似），**默认关闭**，开启时记录抽样方案与种子。

### 4.4 基线在比较中的用途

| 用途 | 规则 |
| --- | --- |
| 地板参照 | 每个报告表必须含 U 与 CB 行；缺行视为报告不完整 |
| IG 的分母基线 | 默认 `IG_CB`（IG 以 CB 为基线，Kümmerer/Tuebingen 惯例），同时报 `IG_uniform` |
| 候选模型及格线 | 见 §8.3（建议，待一级总控确认） |
| 口径自检 | U/CB 的指标值必须落在 §4.1 与解析预期范围内，否则评估脚本判定失败并停止（不产出假成功） |

---

## 5. 观看窗口（viewing duration）

1. UEyes 提供 **1s / 3s / 7s** 三个窗口。官方代码核实：三档是**同一批 7s 自由观看 trial 的注视时间前缀截断**（`get_coordinates(1)`/`(3)`/全量），因此真值满足嵌套关系（7s 注视集 ⊇ 3s ⊇ 1s），并非三组独立实验。
2. **硬性规则：所有指标必须按窗口分别计算、分别报告，禁止混合**。具体禁止：(a) 用窗口 A 的真值评估窗口 B 的预测；(b) 把三个窗口的逐图指标池化成单一均值；(c) 跨窗口平均真值图。
3. 窗口标签 `window ∈ {1s, 3s, 7s}` 是报告表主键的一部分（表 A：模型 × 划分 × 窗口 × 指标）；逐图结果 CSV 每行必须带 window 字段。
4. 跨窗口的数值只允许**描述性对比**（如"IG 随窗口增大而上升/下降"），不得据此做模型选型结论；窗口间真值嵌套导致的相关性在 §7 统计中按同图配对处理。
5. CB、SF 基线按窗口分别构造（§4.2/§4.3）。
6. FiWI（若采纳）只有 5 s 全量自由观看：`window=5s_full`，单独成表，**不与 UEyes 任何窗口同表**。
7. 模型推理本身与窗口无关（静态截图输入）；窗口只改变真值。同一模型输出在三个窗口上分别评估是合法的。

---

## 6. 指标统一口径

本节为规范性（normative）内容。权威依据：Kümmerer, Wallis, Bethge, "Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics", ECCV 2018（arXiv:1704.08615，**已核实全文**，含各指标定义与"模型-图-指标分离"框架）；Kümmerer et al., "Information-theoretic Model Comparison Unifies Saliency Metrics", NeurIPS 2014（IG 的提出文献，**书目引用，全文未核实**——PDF 不可解析）；Bylinskii, Judd, Oliva, Torralba, Durand, "What do different evaluation metrics tell us about saliency models?", IEEE TPAMI 39(2), 2017（arXiv:1604.03605，经 MIT 基准站引文核实 ID 与题录，**全文未核实**——当日代理故障）；Judd et al. 2012（AUC-Judd 出处）；Borji, Sihite, Itti, TIP 2013（sAUC/AUC-Borji 出处，经 MIT 基准站引文核实题录）；pysaliency（Kümmerer 官方实现，`metrics.py` **已核实源码**）；UEyes 官方评估代码（**已核实源码**）。

### 6.0 记号与公共预处理（先于一切指标）

| 约定 | 规定 |
| --- | --- |
| 评估网格 | 每图统一为**原图分辨率** W×H。模型输出若为其他分辨率：双线性插值到 W×H 后**重新归一化**；插值方法、目标尺寸写入配置哈希（与技术方案 §4 一致） |
| 模型概率图 S | 非负、有限、`sum(S)=1`，容差 1e-6（与[验证方案](../validation-plan.md)§1 一致）。log-density 输出（如 DeepGaze）：`S = exp(L − logsumexp(L))`（与[技术方案](../technical-design.md)§7 一致）。**全管线 float64，禁止 8-bit PNG 中间存储参与计算**（Kümmerer 2018 明确指出 JPEG/8-bit 量化影响 AUC 类指标，需直方图均衡补救；本协议直接禁止量化路径） |
| 真值注视点集 | 从原始日志重建（UEyes：Gazepoint CSV，BPOGV=1，坐标映射与越界丢弃规则复现官方管线 §2.1）。默认**计数加权**（每注视点权重 1）；**时长加权**（权重=FPOGD）作为敏感性分析，两者不得混表 |
| 真值模糊图 F | `F = G_σ * F_delta`，F_delta 为注视点 δ 累加图；**σ = 1° 视角**：UEyes `σ = 40 × max(W/1920, H/1200)` px（复现官方换算），FiWI σ ≈ 26 px；F 归一化 `sum(F)=1` |
| 经验注视分布 F̂ | 离散形式：`F̂(p) = w_p / Σw`（w_p 为落在像素 p 的注视权重）；连续形式即 F。IG/NSS/sAUC/AUC 用离散注视点或 F̂；CC/KL/SIM 用模糊 F |
| 对数底 | **log2**（单位 bits/fixation，Tuebingen 基准惯例）；若实现用 ln，结果必须换算或明确标注 nats |
| 数值防护常数 | 混合系数 `λ = 1e-8`；相对地板 `ε_rel = 1e-12`；全零/常数图处理见各指标。防护方案一经冻结写入配置哈希 |

**统一防护原语**（供实现复用）：

```text
to_density(M, λ):                     # 概率图化
    M ← max(M, 0)                     # 负值截断为 0（记录被截断像素数）
    if sum(M) == 0: M ← 1/N（均匀）    # 全零 → 均匀，flag=degenerate
    M ← M / sum(M)
    M ← (1−λ)·M + λ·U                 # 严格正性混合（U=均匀），保证 log 有限
    return M                          # sum 仍为 1
```

### 6.1 IG（Information Gain，bits/fixation，越高越好）

**定义**（Kümmerer et al. NeurIPS 2014；ECCV 2018 已核实文本：`E_p[IG(q)] = Σ_i p_i (log q_i − log p_bl,i)`，其中 q、p_bl 均为概率向量）：

```text
IG(S | B) = (1/|Φ|) · Σ_{(x,y)∈Φ} [ log2 S(x,y) − log2 B(x,y) ]
```

Φ = 该图该窗口的真值注视点集（计数或时长加权时用加权平均）；S、B 均经 `to_density(·, λ=1e-8)` 处理，`sum=1`。

- **基线 B**：默认 CB（§4.2，同数据集×窗口×划分），另报 `IG_uniform`（B=U）。等价形式 `IG = Σ_p F̂(p)·[log2 S(p) − log2 B(p)]`。
- **数值防护**：λ 混合保证 S、B 严格 >0，`log2` 不会遇到 0；若实现选择地板方案 `S ← max(S, ε_rel·max(S))` 再归一化，必须在配置中声明（两种方案数值不同，不可混表）。
- **语义**：相对基线每注视点获得的额外信息量；0 = 与基线持平，负 = 劣于基线。IG 依赖基线选择，**报告时必须写明基线**。
- **注意**：UEyes 官方 eval 的 `infogain` 默认 `rand_map=全零`（等效无真实基线，仅 ε 地板），与本口径**不可比**（§6.8）。

### 6.2 NSS（Normalized Scanpath Saliency，越高越好）

**定义**（Kümmerer 2018 已核实文本："average saliency value of fixated pixels in the normalized (zero mean, unit variance) saliency map"；实现核实自 pysaliency）：

```text
Ŝ = (S − mean(S)) / std(S)          # 均值/标准差为全图像素统计，S 为原始显著图（无需概率归一化）
NSS = (1/|Φ|) · Σ_{(x,y)∈Φ} Ŝ(x,y)   # 在离散注视点上平均（不是模糊图）
```

- **数值防护**：`std(S)=0`（常数图，含均匀基线 U）→ NSS 无定义；**本协议固定：记 0 并打 `constant_map` flag**（pysaliency 对 CC 采用类似 0 值约定；UEyes 官方实现用 `EPS` 兜底，数值上等价于巨大值，弃用）。U 基线的 NSS≈0 是预期 sanity 结果（由 flag 机制而非除零保护产生）。
- 注视点重复落在同一像素时按次数计入（计数加权）或按时长加权。

### 6.3 CC（线性相关系数，越高越好，[−1,1]）

**定义**（Kümmerer 2018 已核实文本：双图归零均值/单位方差后的相关；等价于归一化图的 Pearson 相关；实现核实自 pysaliency）：

```text
CC = Σ_p (F(p) − F̄)(S(p) − S̄) / sqrt( Σ_p (F(p) − F̄)² · Σ_p (S(p) − S̄)² )
```

F = 模糊真值图（§6.0），S = 模型图（仿射不变，无需归一化）。

- **数值防护**：任一侧 std=0 → **固定记 0** 并打 flag（pysaliency 约定：一侧常数另一侧非常数 → 0.0；双常数 → corrcoef 产生 NaN，本协议覆写为 0 + flag）。
- CC 对 F 的模糊 σ 敏感（Kümmerer 2018：最优 CC 图 ≈ 密度与同 σ 高斯卷积）→ σ 必须入配置哈希。

### 6.4 sAUC（shuffled AUC，越高越好，[0,1]）

**定义**（Borji et al. TIP 2013 提出、MIT 基准实现惯例；Kümmerer 2018 已核实其 2AFC/似然比框架：AUC 类指标 = `p_fix/p_nonfix` 的判别性能，sAUC 的 nonfixation 分布取图像无关先验）：

```text
正样本 = 本图真值注视点（离散集合 Φ）
负样本 = 同划分、同窗口内【其他图像】的注视点（shuffled fixations）
分数   = S 在两样本集上的 ROC-AUC（S 为模型图原始值，秩统计量，无需归一化）
```

- **实现约定（冻结项）**：负样本默认取其他图像注视点的**全部**（大划分下可按固定种子抽样 M = 10×|Φ|，记录种子）；ROC 用**全阈值扫描 + 梯形积分**（ties 由梯形法自然处理）。据 Bylinskii et al.（2017）对 MIT 旧实现的转述（该全文未核实，见 §10 #11），其在注视点显著值上采样约 100 个阈值做近似——本协议**不采用**该近似（会引入额外方差）。
- **口径分歧声明**：Bylinskii/MIT 的 sAUC 负样本 = 打乱注视点；Kümmerer 2018 理论框架中 sAUC 的负分布 = 图像无关中心偏置（最优图 = `p_fix/CB`）。两者数值接近但不相同。**本协议默认 shuffled-fixations 实现**；若候选模型作者只公布 center-bias-likelihood 口径分数，标注口径差异，不得直接混表。
- **数值防护**：正或负样本为空 → NaN + 排除该图（计入审计）；S 全常数 → AUC=0.5（ties 全并列的解析结果）。

### 6.5 AUC-Judd 与 AUC-shuffled

**AUC-Judd**（Judd et al. 2012；Kümmerer 2018 已核实：非注视分布为均匀）：

```text
正样本 = 注视像素（F_delta > 0 的像素集合）
负样本 = 同图所有其他像素
分数   = S 的 ROC-AUC（全像素阈值扫描，梯形积分）
```

- **UEyes 官方变体警示**：官方 `eval_heatmaps.auc` 将参考图（8-bit 模糊热图、max 归一化）以 **>0.7 相对阈值**二值化后做全像素 AUC——阈值语义与本口径（注视点存在性）不同。复现官方数字时必须用官方口径并标注；**正式报告用本口径**。
- 大图像全像素 ROC 计算量大：允许对负样本按固定种子均匀抽样（M = 10×正样本数）近似，抽样方案入配置哈希；不抽样为默认。

**AUC-shuffled**：本协议中 **AUC-shuffled ≡ sAUC**（§6.4），全项目统一用术语 `sAUC`，禁止同报告混用两个名字指不同实现。可选增补 **AUC-Borji**（负样本 = 同图均匀随机采样像素点，非注视点集合）作为敏感性行，需显式标注。

### 6.6 KL（Kullback–Leibler 散度，**越低越好**）

**定义**（MIT 基准方向 `KL(F‖S)`；Kümmerer 2018 已核实文本：`KL[e,q] = Σ e_i log(e_i/q_i)`，e=经验图、q=模型图，双方转概率分布；实现与超参数核实自 pysaliency `MIT_KLDiv`）：

```text
KL = Σ_p F(p) · [ log2 F(p) − log2 S(p) ]
F、S 均经 to_density(·) 归一化（sum=1、非负、严格正）
```

- **方向固定为 KL(F‖S)**（真值在前）。**警示**：UEyes 官方 `kldiv` 调用 `scipy.stats.entropy(sal, ref)` = **KL(S‖F)**，方向相反且数值不同（§6.8）。
- **数值防护（默认）**：λ=1e-8 均匀混合（`to_density`），保证 log 有限；`F(p)=0` 的项按 `0·log0 = 0` 跳过。
- **备选（MIT 原超参，对比用）**：`minimum_value=0` + `log_regularization = quotient_regularization = 2.2204e-16`（float32 eps，pysaliency `MIT_KLDiv` 已核实）。两种防护不可混表。

### 6.7 SIM（相似度，可选，越高越好，[0,1]）

**定义**（MIT 基准；pysaliency 已核实）：`SIM = Σ_p min(F̂_d(p), S_d(p))`，双方 `to_density(·, λ=0)`（无混合，纯归一化）。语义 = 分布重叠（l1 距离的补：`SIM = 1 − ½·L1`，Kümmerer 2018 已核实该关系）。默认**关闭**（与 IG/CC 信息重叠），开启时口径同上。

### 6.8 与 UEyes 官方评估实现的已知差异表（防混表资产）

| 指标 | UEyes 官方实现（eval_heatmaps.py，已核实源码） | 本协议口径 | 后果 |
| --- | --- | --- | --- |
| IG | `rand_map` 默认全零 → 第二项恒为 `log2(EPS)`；无真实基线；EPS=float32 eps ≈ 1.19e-7 | IG(S\|CB) 与 IG(S\|U)，λ 混合防护 | **数值不可比**；官方"infogain"实为带常数偏移的 log2 似然 |
| KL | `scipy.entropy(sal+EPS, ref+EPS)` = **KL(S‖F)**；输入未显式概率归一化（scipy 内部各自归一） | **KL(F‖S)**，to_density 防护 | **方向相反** |
| AUC | 参考图 = 8-bit 模糊热图 max 归一后 **>0.7 阈值**二值 | 正样本 = 注视点存在像素 | 阈值语义不同，数值系统性偏移 |
| NSS | mask = 参考图**非零像素**（高斯模糊后几乎全图非零） | 离散注视点 | 语义差异显著（近似全图均值→0） |
| 预测图加载 | 8-bit 灰度 PNG，`max>1` 时除以 max | float64 概率图 sum=1 | 量化 + 归一化约定不同 |
| sAUC | 官方 eval.py **未调用**（代码存在 auc_shuff 但默认 rand_map 全零，不可用） | §6.4 完整定义 | 官方无 sAUC 数字可比 |

**结论**：本项目复现分数与 UEyes 论文表格分数、官方 eval 输出**均不得直接混表**；如需对照官方数字，必须原样运行官方代码并在报告中双口径并列（见 §8.2）。

### 6.9 离散化与插值汇总

- 注视点 → 像素：`px = floor(x)`（左上原点，与技术设计 AOI 坐标约定一致）；越界注视丢弃（复现官方规则）并计数入审计。
- 模型图 → 评估网格：双线性；概率图重采样后**必须重新归一化**（技术方案 §4.5）。
- 模糊 F：高斯核截断半径 ≥ 4σ；边界处理 = 反射（reflect）或零填充后归一化——**二选一冻结**（默认 reflect，与 `scipy.ndimage.gaussian_filter` 默认一致，官方管线同源）。
- CB 上采样：双线性 + 重归一化。

---

## 7. 统计与报告

### 7.1 逐图计算（最小粒度）

每个 `(dataset, split, window, model, image)` 计算一次全部启用指标，落盘逐图 CSV：

```text
dataset, split, window, model, model_version, image_id, category, block,
n_viewers, n_fix, excluded_flag, metric, value
```

聚合与 CI 只从逐图值计算；**禁止**把多图真值/预测拼接后计算"数据集级"指标。

### 7.2 不确定性（最小可实施方案）

- **像素非独立原则**：显著图相邻像素高度相关，任何以像素为样本单位的方差/CI 均无效。**全部不确定性以图像为重采样单位**。
- **图像级 bootstrap（默认）**：对测试划分的图像列表有放回重采样 B=2000 次（固定种子，建议 20260911），每次重算指标均值 → 取 2.5/97.5 百分位为 95% CI。
- **分层 bootstrap（按类别）**：webpage/desktop/mobile/poster 各自内部重采样后合并，避免类别构成波动主导 CI；作为表 C 的默认口径。
- **配对比较（模型 A vs B）**：同图配对差值 `d_i = m_A(i) − m_B(i)`，对 `d_i` 做图像级 bootstrap 得差值 CI；报告差值均值、CI 与胜率 `#{d_i>0}/n`。**两模型 CI 各自不重叠 ≠ 差值显著**，必须用配对差值 CI 判断。
- **多重比较**：同一报告内多模型×多指标的两两检验用 Holm 校正；校正范围在报告中写明。
- **参与者维度（限制声明）**：UEyes 官方聚合真值已跨参与者合并，参与者级重采样需从原始日志按 leave-one-participant-out 重建真值，成本高。**最小方案不做**；报告限制声明"参与者间变异未单独建模，CI 仅反映图像抽样变异"。进阶方案（可选，二期）：随机抽取 20% 参与者剔除后重建真值重算指标，作为稳健性行。
- **混合效应模型（可选进阶，非最小方案）**：`value ~ model * window + (1|image) + (1|category)`；仅当需要跨窗口联合建模时使用，且不得替代逐窗口报告（§5）。

### 7.3 报告表格模板

**表 A（主表）模型 × 划分 × 窗口 × 指标**：

| model | split | window | IG_CB ↑ | IG_U ↑ | NSS ↑ | CC ↑ | sAUC ↑ | AUC-Judd ↑ | KL ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform（基线） | test | 1s | 0.000 [CI] | … | … | … | … | … | … |
| center_bias（基线） | test | 1s | … | … | … | … | … | … | … |
| candidate_X v1.2 | test | 1s | … | … | … | … | … | … | … |
| candidate_X v1.2 | test | 3s | … | … | … | … | … | … | … |
| candidate_X v1.2 | test | 7s | … | … | … | … | … | … | … |

单元格 = `mean [95% CI]`（bootstrap，分层口径注明）；↑/↓ 标注方向；基线行缺失 = 报告不完整。

**表 B 配对差值**：`candidate − center_bias` 与 `candidate − uniform` 的逐窗口差值均值、CI、胜率、Holm 校正后显著性。

**表 C 类别细分**：category × window × 指标（同上格式），用于回答"是否只在某类 UI 上有效"。

**表 D 数据审计**：每 (split × window)：图像数、排除数（low_fixation/low_viewer/空正负样本）、参与者覆盖统计（min/median/max viewers per image）、注视点总数、近似重复簇数与跨划分检查结果、md5 校验结果。

**表 E 配置指纹**：protocol_version、划分文件哈希、模型名/版本/权重哈希、预处理（插值、σ_blur、加权方案）、数值防护（λ、ε）、对数底、基线版本哈希、评估代码版本（Git commit）、bootstrap 参数（B、种子）。**表 E 任一项不同 → 结果不可与旧表同表比较**（§8.1）。

### 7.4 结果记录格式

逐图 CSV + 表 A–E 的 Markdown/JSON 双格式；JSON 字段与[数据契约](../data-contract.md)的结果对象风格对齐（`status/metrics/evidence/config_hash`），所有产物标注 `模型预测` 与"公开数据评估，非游戏 UI 验证"字样（与验证方案 §6 模板一致）。

---

## 8. 可比性条件与成功判据

### 8.1 两结果可同表比较的充要清单

同一行对比成立，当且仅当双方满足**全部**：

1. 同数据集与版本（同 md5）；
2. 同划分文件（同哈希）与同 split（test 对 test）；
3. 同观看窗口；
4. 同评估网格与插值方法；
5. 同真值构造（计数/时长加权、σ_blur、边界处理、越界规则）；
6. 同概率归一化与数值防护常数（λ、ε、对数底）；
7. 同指标定义版本（§6 的 protocol_version）；
8. 同基线版本（IG_CB 的 CB 文件哈希一致）。

任一项不同 → 分表并列 + 显式标注差异项，**不得同表、不得画同轴柱状图**。

### 8.2 文献分数与复现分数

- UEyes 论文表格分数（含勘误后的 Table 2）、官方 eval.py 输出、其他论文报告分数：**一律不得与本项目复现分数混表**（口径差异见 §6.8，勘误见 §2.1）。
- 允许"文献参考值"独立小表：数值 + 出处（论文、表号、版本/勘误状态）+ 口径差异声明。
- 若需与 UMSI++ 等论文模型对比：**用官方发布权重在本协议口径下重跑**，重跑值进表 A，论文值只进参考表。

### 8.3 阶段性成功判据（**建议，待一级总控确认**）

以下判据全部为 R2 研究建议，未经一级总控批准不生效；阈值数字为暂定建议值：

- **S0 管线自检**：均匀基线的 NSS≈0、AUC≈0.5、IG_U≈0（容差由实现前测试确定）；CB 的 IG_CB≈0（定义自洽）。不通过 → 脚本缺陷，禁止出报告。
- **S1 基线有效性**：每个窗口下，CB 的 NSS 与 sAUC 显著高于均匀基线（配对差值 CI 下界 > 0）。不通过 → 检查 CB 构造与真值管线。
- **S2 候选及格线（后端选型核心判据）**：候选模型在 UEyes test 划分、每个窗口下，`IG_CB` 与 `NSS` 的配对差值（vs CB）95% CI 下界 > 0。任一窗口不达标 → 该后端不满足"优于中心偏置"的最低要求，记录并评估替代候选。
- **S3 跨域一致性（定性）**：候选模型在 FiWI（若许可核实通过）与 UEyes-webpage 类别上的相对表现方向一致（相对 CB 的提升同向）。不一致 → 标注"跨数据集不稳定"，不单独否决。
- **S4 相对水平（暂缓定量）**：与 UMSI++（本口径重跑值）的差距是否设阈值（如"IG 达到其 90%"），**留待一级总控裁决**——涉及后端选型策略，本研究不设数。
- **失败动作**：S2 不达标时不得宣称"模型可用于游戏 UI 评审"，只能宣称"在公开 UI 数据上未显著优于中心偏置基线"，并回到后端候选清单。

### 8.4 禁止表述（对外报告红线）

- 不得把任何 AUC/NSS/IG 数值称为"预测准确率""命中率"；
- 不得由公开 UI 结果推出"模型在游戏 UI 上有效/无效"（§0.2）；
- 不得由 A/B 指标差异推出"点击率或任务效率会提高"（与验证方案 §5 一致）；
- 不得省略窗口、划分、基线版本而单独引用一个指标数字。

---

## 9. 边界声明（重申，与 §0 一致）

| 边界 | 内容 |
| --- | --- |
| 数据边界 | 无自采眼动数据；仅公开数据集；UEyes 12.9 GB 未下载（本阶段） |
| 域边界 | 公开数据均为通用 UI（webpage/desktop/mobile/poster），**无游戏 UI**；游戏 UI 预测有效性 = 未验证 |
| 结论边界 | 本协议产出的是"候选后端在公开 UI 数据上的相对表现证据"，仅服务后端选型；不构成产品有效性承诺 |
| 模型边界 | 不训练、不微调；官方预训练权重的数据足迹（如 UMSI++ 见过 UEyes-train）必须声明 |
| 许可边界 | UEyes=CC-BY-4.0（署名）；FiWI 许可未核实前不下载不分发；WebSaliency 不可达；数据不入 Git |

---

## 10. 来源清单与核查状态（核查日期均为 2026-09-11）

| # | 来源 | 用途 | 状态 |
| --- | --- | --- | --- |
| 1 | UEyes GitHub README：<https://github.com/YueJiang-nj/UEyes-CHI2023> | 数据组织、info.csv 字段、代码结构、Gazepoint 日志 | **已核实**（raw README） |
| 2 | Zenodo 8010312：<https://zenodo.org/records/8010312> | 许可 CC-BY-4.0、体量 12.9 GB、md5、发布日期、DOI | **已核实** |
| 3 | UEyes 项目页：<https://userinterfaces.aalto.fi/ueyeschi23> | 论文/附录/权重链接、**Table 2 校准勘误声明** | **已核实** |
| 4 | Aalto 研究门户（UEyes 条目） | 论文元数据、CC BY、发表日期 | **已核实** |
| 5 | UEyes 官方代码：`data_processing/generate_heatmaps.py`、`aggregate_data.py`、`evaluation/eval.py`、`eval_heatmaps.py`、`evaluation/README.md`、`data_processing/README.md`、`saliency_models/UMSI++/README.md` | 真值生成管线、窗口截断语义、σ 换算、官方指标口径、UMSI++ 训练数据 | **已核实**（源码全文） |
| 6 | Kümmerer, Wallis, Bethge, ECCV 2018, arXiv:1704.08615（ar5iv 全文） | 模型/图/指标分离；IG、NSS、CC、KL、SIM、AUC/sAUC 定义与最优图；8-bit 量化警示 | **已核实**（全文文本） |
| 7 | arXiv 检索页（确认 1704.08615 题录与"published at ECCV 2018"） | 文献身份 | **已核实** |
| 8 | pysaliency（Kümmerer 官方实现）`pysaliency/metrics.py` | NSS/CC/KL/SIM 实现、EPS 与 MIT_KLDiv 超参、密度转换、常数图防护 | **已核实**（源码全文） |
| 9 | MIT Saliency Benchmark 首页与数据集页：<http://saliency.mit.edu/>、<http://saliency.mit.edu/datasets.html> | Bylinskii arXiv:1604.03605、Judd 2012、Borji TIP 2013 题录；FiWI/CAT2000/MIT300/MIT1003 参数；σ≈1° 视角约定（Le Meur & Baccino 2013）；站点移交 tuebingen | **已核实** |
| 10 | Kümmerer et al., NeurIPS 2014, "Information-theoretic Model Comparison Unifies Saliency Metrics" | IG 的提出文献 | **书目引用，全文未核实**（PDF 不可解析；定义经 #6 全文与 #8 源码交叉印证） |
| 11 | Bylinskii et al., TPAMI 2017, arXiv:1604.03605 | AUC 变体/指标性质权威综述 | **题录已核实（经 #9），全文未核实**（当日代理故障） |
| 12 | UEyes 论文正文 PDF | 官方划分准则、每图参与者数、任务细节 | **未核实**：web_fetch 不支持 PDF；dl.acm.org 反爬 403；r.jina.ai 代理不可达。**待补**：数据下载后读 info.csv + 论文人工核对 |
| 13 | WebSaliency（`github.com/shenyi-z/WebSaliency`） | 对照数据集候选 | **未核实/不可达**：仓库 404；学术搜索 API 限流。未采纳 |
| 14 | FiWI 官方页（NUS） | FiWI 许可与下载方式 | **未核实**：页面经代理持续故障不可达；FiWI 参数以 #9 清单页为准 |
| 15 | <https://saliency.tuebingen.ai> | 现行基准站（IG 口径运行方） | **未核实**：当日网络不可达（TypeError） |
| 16 | SALICON（salicon.net） | 通用模型预训练数据源说明 | **未核实**：当日网络不可达；仅作背景提及，不入评估 |

引用要求：使用 UEyes 数据/衍生产物的一切对外材料附来源 #1–#4 的完整引用（CC-BY-4.0 署名义务）。

---

## 11. 待裁决项（提交一级总控）

1. **成功判据 S0–S4（§8.3）是否采纳**，尤其 S2 的"每窗口 IG_CB 与 NSS 均须显著优于 CB"是否为后端选型硬门槛；S4 是否设相对 UMSI++ 的定量阈值。
2. **默认真值加权**：计数（默认建议）还是时长加权（UEyes 官方聚合示例用 `-w 1` 时长加权）——影响与官方产物的可比性。
3. **默认 IG 基线**：CB（建议，Tuebingen 惯例）还是同时强制双基线（IG_CB + IG_U）。
4. **sAUC 实现口径**：shuffled-fixations（建议默认）与 Kümmerer center-bias-likelihood 变体是否都实现。
5. **FiWI 是否纳入**：许可需先人工核查官方页/联系作者；不纳入则对照面只剩 UEyes 内部跨类别。
6. **数据下载授权**：UEyes 12.9 GB（含 1,980 张第三方 UI 截图，CC-BY-4.0 允许再分发但本项目不入 Git）何时下载、存放于何处（仓库外数据目录）。
7. **近似重复阈值**（Hamming ≤ 8）与 **排除门槛**（注视 <10、观看者 <3）的冻结值确认。

---

## 附录 A：评估脚本最小伪代码

```text
# 输入：model（可调用，图像→float64 概率图）、dataset_dir、splits.vX.json、
#       cb_{dataset}_{window}_{split}.vX.npz、protocol_config.yaml（σ、λ、ε、加权、种子、B）

for window in [1s, 3s, 7s]:                      # §5：绝不混合
    gt = build_groundtruth(dataset_dir, window)  # 原始日志→注视点集（§6.0；UEyes 复现官方坐标映射与越界丢弃）
    audit(gt)                                    # 表 D：viewers/fixations/排除标记（§3.5）
    for image in split.test_images:              # §3.2 官方划分
        S   = model.infer(image)                 # 原图分辨率，float64
        S   = to_density(resample(S), λ)         # §6.0 防护原语；sum=1 校验（容差 1e-6）
        F   = to_density(gaussian_blur(gt.delta[image], σ=1dva), λ)
        Fh  = gt.discrete[image]                 # 离散注视点（计数或时长加权）
        B   = to_density(upsample(cb[window]), λ)
        rows.append(per_image_metrics(S, F, Fh, B))   # IG_CB, IG_U, NSS, CC, sAUC, AUC-Judd, KL(F‖S)[, SIM]
    rows += baseline_rows(uniform, cb[window])        # §4.4：基线必须过同一管线
report(bootstrap_image_level(rows, B=2000, seed),     # §7.2：图像级重采样，配对差值 + Holm
       tables=[A, B, C, D, E])                        # §7.3；表 E 含全部配置哈希
```

失败行为：md5 不符、S 含 NaN/负值、归一化超差、基线 sanity（S0）不过 → 结构化错误退出，不产出部分成功报告（与技术方案 §10 一致）。

---

*本文档为第一波研究子任务 R2 产出，供一级/二级总控评审；修订需递增版本号并保留旧版可追溯。*
