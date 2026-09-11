# 模型候选与许可核实（R1）

任务代号：R1（第一波·模型候选与许可核实）。核查日期：**2026-09-11**（全部条目同日核查）。
本文档为可公开脱敏内容：不含会话 ID、凭据、本机绝对路径与内部资产信息。

## 0. 核查方法与证据口径

- **只读在线核查**：通过 raw 文件原文、GitHub Releases 资产清单（expanded_assets）、commit Atom feed、Zenodo API、HuggingFace API/模型卡原文取证；GitHub REST API 在本核查期间遭遇共享出口 IP 限流，Issue 状态改经公共代理转发 API JSON 取得。
- **未下载任何权重/数据集二进制，未安装依赖，未运行任何模型推理**。所有体量数字来自远端元数据（已验证）或参数量推算（标注"推断"）。
- **本机复现状态：所有候选一律"未复现"**。文中所有指标均为作者报告值，不构成本项目实测结果。
- 证据分级标注：`已验证` = 本次取得一手来源原文/元数据；`推断` = 由已验证证据合理推导；`未验证` = 缺少可达一手来源（附原因）。
- 许可结论只依据**实际 LICENSE 文件文本 / 官方元数据字段**，不依据 README 口号、徽章或 setup.py 分类器（后者仅作线索并单独注明）。

### 结论速览

| # | 候选 | 输出语义 | 契约 §3 兼容 | 代码许可 | 权重许可 | 综合判定 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | UMSI++ / UEyes（CHI 2023） | 空间显著图（多观看窗口） | 可映射 probability_density，封装成本高 | **无 LICENSE 文件**（未闭合） | 项目页 zip，条款未声明 | 数据集 CC-BY-4.0 闭合可用作基准；模型不选 |
| C2 | foveacast-training（UI 微调 MSI-Net） | 逐图 min-max 归一 [0,1] 显著图 | 兼容（适配层重归一后声明 probability_density） | **MIT 已验证** | MIT + UEyes 署名传递（已验证）；SALICON 上游链残留缺口 | **首选工程候选**（残留缺口见 §6） |
| C3 | DeepGaze IIE / MSDB | **log density**（原生） | 语义最匹配（log_density） | **无 LICENSE，MIT 字段被注释**（未闭合） | Release 无许可声明（未闭合） | 仅限内部研究评估；打包/商用需一级总控裁决 |
| C4 | SeekUI（CHI 2026） | scanpath 坐标序列 + 思考文本 | **不兼容静态热图接口**（时序能力，单独实验线） | 徽章称 MIT 但**无 LICENSE 文件**（矛盾，未闭合） | HF 无模型卡无声明；基座 Qwen2.5-VL-3B **仅非商用**（已验证） | 本轮不选；启动需一级总控裁决 |
| C5 | VLM 解释 / 候选 AOI（调研项） | 文本/框级建议（非密度图） | 不走 predict() 通道；仅 regions(source=agent) 与 review 辅助 | 依所选模型逐一核实（本轮未展开） | 同左 | 可行但严格受限：不得当眼动真值 |

---

## C1. UMSI++ / UEyes（CHI 2023）

### 基本信息与 pin

| 项 | 值 | 状态 |
| --- | --- | --- |
| 代码仓库 | https://github.com/YueJiang-nj/UEyes-CHI2023（分支 main） | 已验证 |
| pin commit | `7bc064175310284c74e64e0c3c2ef264bfbe66e2`（2024-07-16，Atom feed 最新提交） | 已验证 |
| 论文 | Jiang et al., "UEyes: Understanding Visual Saliency across User Interface Types", CHI 2023, DOI [10.1145/3544548.3581096](https://doi.org/10.1145/3544548.3581096) | 已验证（项目页） |
| 项目页 | https://userinterfaces.aalto.fi/ueyeschi23/ | 已验证 |
| 数据集 | Zenodo record [8010312](https://zenodo.org/records/8010312)，DOI 10.5281/zenodo.8010312 | 已验证 |

### 输入/输出语义与契约兼容性

- 仓库结构（jsDelivr 全量文件清单，已验证）：`data_processing/`、`evaluation/`（热图与 scanpath 指标实现）、`saliency_models/UMSI++/`、`scanpath_models/{DeepGaze++, PathGAN++}/`。
- UMSI++ 以 Jupyter notebook 形态提供（`UMSI++_train.ipynb` / `UMSI++_eval.ipynb`），训练/评估目标为 UEyes 的 fixation map 与 heatmap（1s/3s/7s 三窗口）。输出语义：**空间显著性图（概率密度类）**，具体归一化方式未逐行核实（时间盒收口，标注未验证）。
- 契约兼容性：语义可声明为 `probability_density`，但**无标准化 describe()/predict() 接口**，notebook 形态 + TF1 老栈使适配封装成本显著高于 C2/C3；多窗口能力超出第一版消费范围（契约 §3 只消费空间密度）。
- DeepGaze++/PathGAN++ 为 scanpath 模型，属时序能力，不在第一版范围。

### 许可四件套

| 层 | 结论 | 证据 |
| --- | --- | --- |
| 代码 LICENSE | **仓库无任何 LICENSE 文件 → 未闭合（默认保留所有权利）** | jsDelivr 全量文件清单无许可文件；raw `LICENSE`、`LICENSE.md` 均 404（已验证） |
| 权重 | `umsi++.hdf5` 不在仓库内（子目录 README 要求自行放入 `weights/`）；项目页提供 [model_weights.zip](https://userinterfaces.aalto.fi/ueyeschi23/model_weights.zip)，**许可未声明；体量未验证（按边界不下载）** | 项目页 + 子目录 README 原文（已验证渠道，许可未验证） |
| 骨干/预训练 | Keras 自定义实现（`src/xception_custom.py`、`dcn_resnet_new.py`、`singleduration_models.py`、`attentive_convlstm_new.py`）；具体骨干与预训练权重来源未核实（时间盒收口） | 文件清单（已验证存在）；骨干细节未验证 |
| 训练数据 | **UEyes 数据集：CC-BY-4.0，允许商用，须署名（引用 Jiang et al. 2023）** | Zenodo API 元数据 `license.id = "cc-by-4.0"`（已验证） |

### 数据集细节（评估基准价值，供 R2 协同）

- 单文件 `UEyes_dataset.zip`：**12,912,746,325 B（≈12.9 GB）**，md5 `c2d53e6af0a47e1f459416d6839ec2c1`（Zenodo API，已验证）；解压后磁盘占用约 25 GB（C2 仓库 data/README 实测记录，二手已验证）。
- 内容：1,980 张 UI 截图（web/desktop/mobile/poster 各 495）、62 名参与者眼动、1s/3s/7s 三窗口的 fixmaps/heatmaps/overlay/scanpaths、原始 Gazepoint 日志 554 份 CSV；上游划分 train/test = 1,872/108（`image_types.csv`）。（C2 仓库 data/README 对包内结构的逐项清点，二手已验证；本项目未下载核对。）
- **边界声明：UEyes 为普通 UI（网页/桌面/移动/海报），无游戏 UI 类别。其上的一切指标不能证明游戏 UI 预测有效**（与 README/来源记录既有口径一致）。

### 依赖与运行要求

- conda `environment.yaml`（已验证原文）：Python 3.7.9、tensorflow-gpu **1.14.0**（cudatoolkit 9.0 + cuDNN 7.6.5）、keras 2.3.1、jupyter/notebook 6.1.4。
- CUDA 9.0 工具链与现代 GPU/驱动基本不兼容（**推断**）；CPU 理论可行但性能未验证。老栈是本候选最大工程障碍。

### 作者报告 vs 本机复现

- 作者报告：UEyes 论文称 UMSI++ 在多类 UI 上优于所比较的 SOTA 显著性模型（论文正文本轮未拉取，仅 README/项目页摘要级信息，标注未验证细节）。
- 本机复现：**未复现**（未安装、未推理）。

### 来源清单（核查日期 2026-09-11）

- https://github.com/YueJiang-nj/UEyes-CHI2023 （README、UMSI++ 子 README、environment.yaml、文件清单、commit feed）
- https://zenodo.org/records/8010312 （API 元数据：许可/体量/md5/DOI）
- https://userinterfaces.aalto.fi/ueyeschi23/ （项目页：权重 zip 渠道、勘误说明）

---

## C2. foveacast-training（UI 微调 MSI-Net）——优先工程候选

### 基本信息与 pin

| 项 | 值 | 状态 |
| --- | --- | --- |
| 仓库 | https://github.com/khawkins98/foveacast-training（分支 main） | 已验证 |
| pin commit | `65209a2df37c1330dfd0cd93cce8d6da50427d27`（2026-05-07，Atom feed 最新提交） | 已验证 |
| 发布 | Release **v0.2.0**（2026-04-18），15 个资产（6 ONNX + 6 parity/quality JSON + 源码包） | 已验证（expanded_assets 清单） |
| 论文（架构上游） | Kroner et al. 2020, Neural Networks 129:261-270, DOI [10.1016/j.neunet.2020.05.004](https://doi.org/10.1016/j.neunet.2020.05.004) | 已验证（README 引用） |

### 输入/输出语义与契约兼容性

`src/foveacast_training/msinet.py` docstring + 代码（已验证原文）：

- **输入**：`(N, 3, H, W)` float32，RGB，值域 `[0, 255]`，H/W 须被 8 整除；均值减除在图内完成（Kroner 特例均值 `(103.939, 116.779, 123.68)` 按 RGB 序），调用方不得预处理。发布 ONNX 推理尺寸 **240×320**（README quickstart）。
- **输出**：`(N, 1, H, W)` float32，**逐图 min-max 归一化到 [0,1]** 的相对显著图——**不是概率密度，也不是 log density**（不保证 sum=1）。
- 契约兼容性判断：**兼容，需适配层**。predict() 封装简单（onnxruntime `InferenceSession`，README 五行 quickstart 已验证存在）；适配层须做 sum=1 重归一化后声明 `probability_density`，并在 profile/limitations 记录两点：① 原始输出为逐图 min-max 相对量，**跨图绝对强度不可比**（本项目 AOI mass/relative_density 为同图内统计，A/B 为同配置比较，均不受影响）；② 240×320 分辨率上限对游戏 UI 小元素（小字、小图标）可能无法解析（README Limitations 自述，已验证）。
- describe() 所需能力/设备/许可字段全部可从已验证材料构造。

### 许可四件套

| 层 | 结论 | 证据 |
| --- | --- | --- |
| 代码 LICENSE | **MIT，Copyright (c) 2026 Ken Hawkins —— 闭合** | raw `LICENSE` 全文已验证（非 classifier/口号）；`pyproject.toml` `license={text="MIT"}` 一致 |
| 发布权重（微调 ONNX） | 仓库 README 声明：产物 = MIT 代码，**下游使用必须携带 UEyes 署名（引用 Jiang et al. 2023，CC-BY-4.0 传递义务）** —— 声明链自洽 | README "Licences, in one place" 原文（已验证）；属作者自声明，法律文本层面权重无独立 LICENSE 文件（注记） |
| 骨干（MSI-Net 上游） | 代码 **MIT，Copyright (c) 2019 Alexander Kroner —— 闭合**；HF 权重卡元数据 `license: mit` —— 闭合 | raw `alexanderkroner/saliency` `LICENSE.md` 全文已验证（pin `bd81e5825cf0548dc5bc7464bda1a4b035be2f4c`，2024-05-28）；HF `alexanderkroner/MSI-Net` README frontmatter 已验证 |
| 骨干的骨干（VGG16 初始化 + SALICON 训练） | encoder 由 **VGG16 ImageNet 预训练**初始化、在 **SALICON**（鼠标追踪代理注视）上训练（HF 卡原文，已验证）。Keras VGG16 代码 Apache-2.0（文件头已验证）、权重由 keras-applications 公共存储分发；**原始 VGG 权重条款（通称 CC-BY-4.0）未从一手来源核实**；**SALICON 条款未核实：salicon.ai DNS 解析失败、salicon.net 抓取失败，二手检索来源称"仅非商业研究用途"——未验证** | 见 §6 缺口 G3/G4 |
| 训练数据 | UEyes，CC-BY-4.0（闭合，须署名） | Zenodo API（已验证，同 C1） |

README 自称 "Nothing here depends on closed-source or non-commercial components"——在 SALICON 条款未经一手核实前，**该断言不采信为结论**，仅记录为作者立场。

### 权重/数据获取渠道与体量

- 微调 ONNX（推理即用，**无需训练环境**）：GitHub Releases v0.2.0 直接下载，含 sha256（已验证清单）：

| 资产 | 体量 | sha256（前 16 位） |
| --- | --- | --- |
| foveacast-v3-1s-fp16.onnx | 53.9 MB | `4b9fdc2734e36c61` |
| foveacast-v3-1s-int8.onnx | 30.3 MB | `3d94ce75c9500b61` |
| foveacast-v3-3s-fp16.onnx | 53.9 MB | `842a23f97908d146` |
| foveacast-v3-3s-int8.onnx | 30.3 MB | `e02e3071af18e11d` |
| foveacast-v3-7s-fp16.onnx | 53.9 MB | `cf66388dc6fe5db4` |
| foveacast-v3-7s-int8.onnx | 30.3 MB | `5ebbe008d7a693cc` |

  完整 sha256 见 Release 页；各资产带 `.parity.json`/`.quality.json` 兄弟文件。FP16 为主推精度（作者称指标与 FP32 四位小数一致，作者报告）。
- stock MSI-Net 权重（对照基线用）：HF `alexanderkroner/MSI-Net` TF SavedModel，导入脚本 pin revision **`MSINET_HF_REVISION = d950b35945db961ae63f84bc2b23f6bd578d0b8f`**（代码常量已验证）；约 100 MB 网络传输（README 声称，未验证）；导入需一次性安装 TensorFlow（约 500 MB，README 声称）。
- 训练数据 UEyes：12.9 GB zip + 解压约 25 GB 磁盘（同 C1；`data/fetch.sh` 渠道，支持 sha256 校验变量）。**仅复现训练/本地评估需要；纯推理不需要。**

### 依赖与运行要求

- **推理**：Python ≥3.12 生态、`onnxruntime ≥1.17`（CPU 即可，作者已验证 PyTorch↔ORT CPU parity）、numpy、pillow；ONNX opset 17；**无 CUDA 硬需求**（下游 Foveacast 在浏览器经 onnxruntime-web 运行，README 已验证）。
- 训练/复现（可选）：torch ≥2.1、torchvision ≥0.16、Apple MPS 或 CUDA GPU；作者记录 M4 MPS 全量微调约 3.5 小时/30 epochs（作者报告，未复现）。

### 作者报告 vs 本机复现

作者报告（UEyes test split 108 张，held-out；**本机一律未复现**）：

| 窗口 | 指标 | 微调后 | stock（SALICON-only） | 作者称改善 |
| --- | --- | --- | --- | --- |
| 3s | CC | 0.7068 ± 0.105 | 0.4934 ± 0.094 | +43% |
| 3s | KLD | 0.6574 ± 0.210 | 1.1682 ± 0.246 | −44% |
| 3s | NSS | 2.2879 ± 0.605 | 1.5776 ± 0.451 | +45% |
| 1s/7s | CC/NSS/KLD | — | — | +60%/+64%/−36%（1s）、+25%/+28%/−44%（7s）（commit 03ec62a 信息） |
| INT8 | 质量回归 | ≤3%（KLD 最敏感 +1.7~2.6%，CC/NSS <0.1%） | — | — |

作者自述局限（README Limitations，已验证）：训练分布为 2020–2022 西方语言桌面/移动 UI，对深色模式、非拉丁文本、新设计范式泛化未知；62 人特定人群；240×320 分辨率上限；非点击预测；单图无任务上下文。**游戏 UI 不在其训练分布内——效果必须由本项目基准评估（R2 协议）实测，不得引用上表作为游戏 UI 有效性证据。**

### 来源清单（核查日期 2026-09-11）

- https://github.com/khawkins98/foveacast-training （README、LICENSE、pyproject.toml、msinet.py、import_msinet_weights.py、data/README.md、commit feed）
- https://github.com/khawkins98/foveacast-training/releases/tag/v0.2.0 （expanded_assets 资产清单 + sha256）
- https://github.com/alexanderkroner/saliency （LICENSE.md、README、commit feed）
- https://huggingface.co/alexanderkroner/MSI-Net （模型卡 raw：license:mit、VGG16 初始化、SALICON 训练、局限声明）

---

## C3. DeepGaze IIE / DeepGaze MSDB

### 基本信息与 pin

| 项 | 值 | 状态 |
| --- | --- | --- |
| 仓库 | https://github.com/matthias-k/DeepGaze（分支 main，pip 包名 `deepgaze_pytorch`，setup.py VERSION 1.2.1） | 已验证 |
| pin commit | `c7db17e2d1d7ea6468ffdee2cfaddf141095dcff`（2026-08-25，"Adapting DeepGaze MSDB to new datasets (#26)"） | 已验证 |
| 许可 Issue | [#15 "Question about License"](https://github.com/matthias-k/DeepGaze/issues/15)：**state=open，0 条评论，维护者从未回应**；创建于 2023-10-11，正文询问"是否愿意添加允许商业使用的许可" | 已验证（GitHub API JSON，经代理转发，2026-09-11 实时） |

### 输入/输出语义与契约兼容性

`README` + `deepgaze2e.py` + `deepgazemsdb.py` 源码（已验证原文）：

- **DeepGaze IIE**：`model(image_tensor, centerbias_tensor) -> log_density_prediction`。输入 image `(B,3,H,W)` [0,255]；centerbias 为**显式输入**的 log density `(B,H,W)`（官方提供 MIT1003 先验模板 `centerbias_mit1003.npy`，或用全零均匀先验）。输出 **log density**。训练域：MIT1003，35 px/dva，长边约 1024（README 注记：输入需按其观看条件缩放）。
- **DeepGaze MSDB**：`model(image, centerbias, pixel_per_dva=…, dataset=MSDBDataset.X|None) -> log density`。**pixel_per_dva 为必填参数**（显示物理条件，不能从截图推断——与 technical-design §5 的"不猜测"原则一致；固定实验配置须标记为实验假设）；`dataset=None` 用跨数据集平均参数泛化。另有 `add_dataset()`/`msdb_adaptation`：**只训练 13 个 per-dataset 标量、骨干与显著网络冻结**的新数据集适配通道（源码已验证）——对未来游戏 UI 域适配是低成本路线（工程线索，未实测）。
- 契约兼容性：**五个候选中语义最匹配**——输出原生即 `log_density`，中心偏置为显式版本化输入，直接对应契约 §3 的 PredictionResult 语义声明与 §4 的 profile 记录要求；technical-design §7 的 `P = exp(L - logsumexp(L))` 转换即按此语义设计。MSDB 的 ppd 需求对应 BackendInfo 设备/配置要求字段。

### 许可四件套

| 层 | 结论 | 证据 |
| --- | --- | --- |
| 代码 LICENSE | **仓库无 LICENSE 文件；setup.py 中 MIT classifier 与 `license='MIT'` 字段均处于注释状态 → 未闭合（默认保留所有权利）** | jsDelivr 全量文件清单（无许可文件）+ setup.py raw 原文（均已验证；与 sources-and-decisions 既有记录一致） |
| 权重 | Release 资产：`deepgaze2e.pth`（v1.0.0，**400 MB**，2021-06-01）、`centerbias_mit1003.npy`（8 MB）、`deepgazemsdb.pth`（v1.2.0，112 KB，sha256 `52b47112035744ed0da95593fc82564676e3da536310face074ed25de6eee6c7`）——**Release 页与文件本身均无任何许可声明 → 未闭合** | expanded_assets 清单（已验证） |
| 骨干（IIE） | 4 骨干混合 ensemble（每骨干 3 实例 × 10 折，deepgaze2e.py 已验证）：① **RGBShapeNetC** = ResNet-50（Stylized-ImageNet+ImageNet 训练），权重从 bitbucket `robert_geirhos/texture-vs-shape-pretrained-models` 固定 URL 下载（源码已验证）；`rgeirhos/texture-vs-shape` GitHub 仓库 **main/master 均无 LICENSE 文件（404 已验证；仓库 >50MB 无法 jsDelivr 全量列表）→ 权重条款未声明，缺口**；② **RGBEfficientNetB5** = 仓库内 vendored `efficientnet_pytorch`（lukemelas 移植，**vendored 副本无许可头**，model.py 已验证）；上游 LICENSE = **Apache-2.0（已验证）**；`from_pretrained('efficientnet-b5')` 运行时自动下载 ImageNet 权重；③④ **RGBDenseNet201 / RGBResNext50** = `torch.hub.load('pytorch/vision:v0.6.0', …, pretrained=True)`（源码已验证）→ torchvision 框架 BSD-3-Clause，ImageNet 预训练权重随框架分发（其独立条款未见明文，见 §6 G8） |
| 骨干（MSDB） | **CLIP ResNet50x64**（`pip install git+https://github.com/openai/CLIP.git`，源码报错提示已验证）：**CLIP LICENSE = MIT, Copyright (c) 2021 OpenAI（全文已验证）**，权重自 OpenAI 公共 URL；**DINOv2 ViT-B/14**：`torch.hub.load('facebookresearch/dinov2:6a62615', …)`（**pin commit 已在源码验证**）：**DINOv2 代码与权重 = Apache-2.0（LICENSE 全文 + README "License" 节双重已验证**；README 中非商用条款仅适用于 XRay-DINO/Cell-DINO 生物变体，与本候选无关） |
| 训练数据 | IIE：MIT1003；MSDB：MIT1003、CAT2000、COCO-FreeView、Daemons、Figrim 五数据集（MSDBDataset 枚举已验证）。**各数据集使用条款本轮未逐一核实（时间盒）→ 未验证** | 见 §6 G5 |

### 体量与运行要求

- 下载体量：IIE = deepgaze2e.pth 400 MB + centerbias 8 MB + 4 骨干预训练权重运行时自动下载（torchvision densenet201/resnext50 约 80–180 MB 级、EfficientNet-B5 约 120 MB 级、ShapeNetC ResNet-50 约 100 MB 级——**均为推断，未验证**）。MSDB = 头部权重仅 112 KB，但骨干为 **CLIP RN50x64（约 1.1–2.2 GB，按公开参数量推断）+ DINOv2 ViT-B/14（约 330 MB，推断）**。
- 依赖：`boltons, numpy, torch, torchvision, setuptools`（setup.py 已验证）；MSDB 额外 `clip`（git 安装）、`einops`，且 torch.hub/权重下载需运行时联网。
- 计算：README 示例 `DEVICE='cuda'`；IIE 为 4 骨干 × 30 组件混合；**MSDB 每图执行 10 次多尺度骨干前向**（5 个 pixel_per_dva 尺度 + 5 个尺寸尺度，源码 `_PIXEL_PER_DVA_SCALES`/`_SIZE_SCALES` 已验证）→ 显著重于 C2。CPU 可行性未验证。

### 作者报告 vs 本机复现

- 作者报告：IIE（Linardos et al. 2021, arXiv:2105.12441）与 MSDB（Kümmerer et al., ICCV 2025）论文指标本轮未拉取正文；README 不含基准数字表。**未复现**。
- 本机复现：**未复现**（未安装、未推理）。

### 来源清单（核查日期 2026-09-11）

- https://github.com/matthias-k/DeepGaze （README、setup.py、deepgaze2e.py、deepgazemsdb.py、features/{shapenet,clip_resnet,dino,densenet,efficientnet,resnext}.py、文件清单、commit feed）
- https://github.com/matthias-k/DeepGaze/issues/15 （API JSON：open、0 评论）
- https://github.com/matthias-k/DeepGaze/releases （v1.0.0 / v1.2.0 资产与 sha256）
- https://github.com/openai/CLIP （LICENSE 全文）
- https://github.com/facebookresearch/dinov2 （LICENSE 全文 + README License 节 + torch.hub pin）
- https://github.com/rgeirhos/texture-vs-shape （master README；LICENSE 404）
- https://github.com/lukemelas/EfficientNet-PyTorch （LICENSE 全文）

---

## C4. SeekUI（CHI 2026，scanpath 单独实验线）

### 基本信息与 pin

| 项 | 值 | 状态 |
| --- | --- | --- |
| 仓库 | https://github.com/YueJiang-nj/SeekUI-CHI2026（分支 main） | 已验证 |
| pin commit | `404a3074321b1c0a2789a344c92b1f0a0dbaaf64`（2026-07-11，Atom feed 最新提交） | 已验证 |
| 论文 | "SeekUI: Predicting Visual Search Behavior on Graphical User Interfaces with a Reward-Augmented Vision Language Model", CHI 2026, DOI [10.1145/3772318.3791178](https://dl.acm.org/doi/full/10.1145/3772318.3791178)（正文未拉取） | 部分验证（README 徽章/链接） |
| 权重 | HF `sushizixin1/SeekUI`（RL 后）与 `sushizixin1/SeekUI_sft`（SFT 阶段），2026-03 发布 | 已验证（HF API） |

### 输入/输出语义与契约兼容性

README Quick Start 代码（已验证原文）：

- **输入**：GUI 截图 + 图像宽高（写入 prompt）+ **目标元素文字**（如 `"WOMEN"`）。
- **输出**：`<think>` 推理文本 + `<answer>` **scanpath 像素坐标序列** `[x1,y1] [x2,y2] …`（注视路径，含时序顺序语义；第一阶段 fixations 带 duration 字段见数据格式）。
- 契约兼容性判断：**不满足静态热图后端接口**——输出为 scanpath 序列（时序/路径能力），契约 §3 明确"时序能力不属于第一版消费范围"；orchestration-plan 亦规定任务搜索模型单独研究、不与静态热图混用、不得用任务描述人为重绘静态热图。若未来立项：describe() 须声明 scanpath capability，走独立后端 profile 与独立评估，不产生 probability_density/log_density。

### 许可四件套

| 层 | 结论 | 证据 |
| --- | --- | --- |
| 代码 LICENSE | **README 徽章声称 MIT，但仓库根目录无 LICENSE / LICENSE.md / LICENSE.txt（raw 全部 404）→ 声明与文件矛盾，未闭合**。（仓库 >50MB，jsDelivr 拒绝全量列表，未能穷举子目录——残余不确定性已注记） | raw 404 ×3（已验证）；徽章仅线索 |
| 权重 | HF `sushizixin1/SeekUI`：**无 README/模型卡文件，API tags 无 license 字段 → 许可未声明**；safetensors BF16 参数量 **3,754,622,976**，与 `Qwen/Qwen2.5-VL-3B-Instruct` 参数量完全一致 → 基座为 Qwen2.5-VL-3B（**已验证推断**）；仓库总存储 15.0 GB（API usedStorage，含 training_args.bin 等；纯权重约 7.5 GB BF16，推断） | HF API JSON（已验证） |
| 基座模型 | **Qwen RESEARCH LICENSE AGREEMENT（2024-09-19 版）：仅非商用（"FOR NON-COMMERCIAL PURPOSES ONLY"，Non-Commercial = 仅研究/评估），商用须向阿里云另行请求许可；再分发须附协议与声明** → SeekUI 微调权重为其派生物，商用路径受限 | HF `Qwen/Qwen2.5-VL-3B-Instruct` cardData `license_name: "qwen-research"` + LICENSE 全文（已验证） |
| 训练数据 | **VSGUI** 眼动 scanpath 数据集：图像与标注经 Google Drive 链接分发（条款未知）；预处理训练集 HF `sushizixin1/vsgui_train_seekui_qwen2_5_explanation`（2.6 GB，sha `3973120f…`）：**API tags 无 license → 未声明**；explanation 数据由 Qwen2.5-VL-72B-Instruct 生成（其自身许可未核实），GPT-5 版数据"coming soon"未发布 | data/README.md 原文 + HF API（已验证渠道，许可未验证） |

### 依赖与运行要求

- conda Python 3.10.12；`vllm 0.7.1`、`transformers 4.55.0→4.51.1`（README 内先装后降级，注记其环境脆弱性）、`trl 0.15.0`、`deepspeed 0.16.2`、`flash-attn 2.7.4`（**需 CUDA 编译 + CUDA GPU**，README note 已验证）、`qwen_vl_utils` 等（README 安装清单已验证）。
- 推理最低显存约 7.5 GB+（BF16 3.75B 参数，推断）；训练（SFT+GRPO）为多卡级（未验证细节）。

### 作者报告 vs 本机复现

- 作者报告：论文称两阶段（指令微调 + 奖励增强 RL）优于基线；具体指标未拉取（ACM DL 正文未取得）。**未复现**。
- 本机复现：**未复现**。

### 结论

三层许可全部未闭合（代码 LICENSE 缺失且与徽章矛盾 / 权重无声明 / 基座仅非商用）+ 数据许可不明。**本轮仅作为独立实验线存档，不选用**；若未来以非商用研究目的启动，需一级总控裁决（并建议先向作者取得书面许可澄清）。

### 来源清单（核查日期 2026-09-11）

- https://github.com/YueJiang-nj/SeekUI-CHI2026 （README、data/README.md、LICENSE 变体 404、commit feed）
- https://huggingface.co/sushizixin1/SeekUI （API 元数据）
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct （API cardData + LICENSE 全文）
- https://huggingface.co/datasets/sushizixin1/vsgui_train_seekui_qwen2_5_explanation （API 元数据）

---

## C5. 调研项：视觉语言模型（VLM）用于解释与候选 AOI

**定位（红线先行）**：本路线**不是显著性后端**，不产生概率图；**不得把 VLM 猜测的注视点/区域当作眼动真值、评估答案或热图来源**（与 orchestration-plan §4 一致）。两个合法用途：

1. **候选 AOI 建议**：VLM/GUI grounding 模型输出元素框与语义标签 → 映射为契约 §2 `regions`（`source: "agent"`、`status: "candidate"`，展示边框，指标与结论标注基于候选边界，人工确认后升级 `confirmed`）。
2. **语义解释辅助**：作为 technical-design §8 Agent 评审的输入侧素材；review.json 中 VLM 观察只能标 `observed`/`inferred`/`unverified` 证据类型，**不得标 `computed`**，不得冒充模型数值。

### 可行性证据（一手，已验证）

- **arXiv 2604.26352（任务书参考链接，可访问）**：*"UIGaze: How Closely Can VLMs Approximate Human Visual Attention on User Interfaces?"*（Min Song, Yoonseong Lee, Yeonhu Seo；2026-04-29 提交；cs.HC；arXiv 页面标示 CC BY 4.0）。摘要级结论（已验证原文）：在 **UEyes 全量 1,980 张**上以零样本坐标预测管线评估 **9 个 SOTA VLM**（1,980 图 × 9 模型 × 3 runs × 3 窗口），坐标经高斯模糊转显著图后与真人眼动比 CC/SIM/KLD；结果 **仅中等一致（moderate alignment）**，且**随观看窗口变长而提高**——提示 VLM 捕捉的是探索性观看模式而非初始注视，跨 UI 类型差异显著。论文自称代码/预测/结果公开，但摘要页未列链接 → **仓库未验证**。
  - 对本项目的直接含义：① VLM 热图化只能作弱代理信号，**质量不足以替代专用显著性模型**（C2/C3）承担数值计算；② "窗口越长越一致"与 UEyes 多窗口基准（R2 协议）可交叉引用；③ 支持把 VLM 限定在候选 AOI + 解释的定位。
- **SeekUI（C4）本身即"VLM 输出结构化注视数据"的实证**：证明该形态技术可行（坐标序列 + 思考文本），但其许可状况不支持本轮选用。
- 候选 AOI 的具体模型路线（GUI grounding：UGround、OS-Atlas、SeeClick 等；或通用 VLM 检测）**本轮未完成许可核实**（两次定向检索超时，时间盒收口）→ 标注**未验证**；若立项，须先按本文档同一标准（LICENSE 实际文本 + 权重卡 + 基座 + 数据四件套）补核。
- 部署形态边界：云端 VLM API 涉及截图上传，须遵守 technical-design §10——外部 API 路线需明确数据上传授权，未授权的游戏截图不得上传；本地开源权重 VLM 则逐模型核实许可（例：Qwen2.5-VL-3B 为仅非商用研究许可，已验证，见 C4）。

### 来源清单（核查日期 2026-09-11）

- https://arxiv.org/abs/2604.26352 （摘要页原文；HTML 版 https://arxiv.org/html/2604.26352v1 存在于该页链接）
- https://github.com/YueJiang-nj/SeekUI-CHI2026 （同 C4）
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/LICENSE （同 C4，VLM 许可样例证据）

---

## 6. 许可缺口汇总（单列）

### 6.1 许可闭合、可用（附义务）

| 资产 | 许可 | 义务 |
| --- | --- | --- |
| UEyes 数据集（Zenodo 8010312） | CC-BY-4.0（Zenodo API 已验证） | 署名：引用 Jiang et al. 2023；允许商用 |
| foveacast-training 代码 | MIT（LICENSE 全文已验证） | 保留版权声明 |
| foveacast v0.2.0 发布 ONNX | 作者声明 MIT + UEyes 署名传递（README 原文） | 下游携带 Jiang et al. 2023 引用 |
| MSI-Net 上游代码（alexanderkroner/saliency） | MIT（LICENSE.md 全文已验证） | 保留版权声明 |
| MSI-Net HF 权重（alexanderkroner/MSI-Net） | 模型卡 `license: mit`（已验证） | — |
| DINOv2 代码与权重（facebookresearch/dinov2） | Apache-2.0（LICENSE 全文 + README 已验证） | Apache 义务 |
| OpenAI CLIP 代码 | MIT（LICENSE 全文已验证） | 保留版权声明 |
| UIGaze 论文文本 | arXiv 页面标示 CC BY 4.0 | 署名（其代码仓库未验证） |

### 6.2 未闭合缺口清单（G 编号供后续引用）

| # | 缺口 | 影响范围 | 状态与证据 |
| --- | --- | --- | --- |
| G1 | **DeepGaze 代码与权重无任何许可**：仓库无 LICENSE；setup.py MIT 字段被注释；Issue #15（2023-10-11 请求商用许可）open、0 评论、维护者未回应 | C3 全部（IIE/MSDB/III 代码 + deepgaze2e.pth + deepgazemsdb.pth + centerbias 文件） | 已验证（三处一手证据）。默认著作权保留所有权利：**不得打包、不得再分发**；内部研究评估是否可行需一级总控裁决（或取得作者书面授权） |
| G2 | ShapeNetC 骨干权重条款未声明：texture-vs-shape 仓库无 LICENSE（main/master 404），权重实际托管于 bitbucket | C3-IIE（4 骨干之一） | 已验证缺文件；bitbucket 侧条款未核查（未验证） |
| G3 | **SALICON 训练数据条款未核实**：官网 salicon.ai DNS 解析失败、salicon.net 抓取失败；二手检索来源称"仅非商业研究用途、禁止再分发、图像源自 MS-COCO" | 一切"SALICON 预训练"权重链：**MSI-Net HF 权重、foveacast 微调 ONNX（自 stock 权重微调而来）**、kroner 仓库自动下载权重 | **未验证（一手站点不可达）**。foveacast README 的"无非商用依赖"断言在 G3 核实前不采信。商用发布前必须补核或取得书面澄清 |
| G4 | 原始 VGG16 预训练权重条款（通称 Oxford VGG CC-BY-4.0）未从一手来源核实；keras-applications 分发副本未见独立许可声明 | MSI-Net 权重链（encoder 初始化） | 未验证（推断为 CC-BY-4.0，署名类义务，商用障碍低但需确认） |
| G5 | DeepGaze 训练数据集条款未逐一核实：MIT1003、CAT2000、COCO-FreeView、Daemons、Figrim | C3 权重链 | 未验证（时间盒收口） |
| G6 | **UMSI++ 代码无 LICENSE + model_weights.zip 权重条款未声明** | C1 模型 | 已验证缺文件/缺声明。数据集本身闭合（CC-BY-4.0），不受影响 |
| G7 | **SeekUI 三层缺口**：代码无 LICENSE（与 MIT 徽章矛盾）；HF checkpoint 无模型卡无 license 标签；基座 Qwen2.5-VL-3B = Qwen RESEARCH LICENSE **仅非商用**；VSGUI 数据（Drive + HF）无许可声明 | C4 全部 | 已验证（矛盾与缺失均一手取证；Qwen 许可全文已验证） |
| G8 | ImageNet 预训练骨干的通用残留风险：torchvision（BSD-3 框架）随附 ImageNet 预训练权重、Keras VGG16 权重等，其训练数据 ImageNet 的使用条款面向研究/教育，业界对"经 ImageNet 训练的权重"商用口径不一 | C2/C3 权重链深层 | 未验证（无明文单一来源）；商用发布时统一记录该残留风险 |

### 6.3 判定汇总

- **许可闭合可用**：UEyes 数据集（基准数据）；foveacast-training 代码 + v0.2.0 ONNX 权重（**用于内部研究与评估**；若走向对外分发/商用，先关闭 G3/G4/G8）；MSI-Net 上游代码与 HF 权重；DINOv2、CLIP（作为组件许可闭合，但所在整体 C3 未闭合）。
- **需一级总控裁决**：① DeepGaze（G1）任何形式的使用边界（含"仅内部评估是否可接受"）；② SALICON 链（G3）对 foveacast 权重商用性的影响与处置策略（补核 / 作者书面确认 / 仅限非商用）；③ SeekUI（G7）若启动非商用研究线；④ ImageNet 类残留风险（G8）的发布披露口径。
- **不得打包或分发**（在裁决前）：DeepGaze 代码与权重、UMSI++ 代码与权重、SeekUI 代码与权重。

---

## 7. 推荐短名单与理由（工程可行性 × 许可 × 静态热图范围匹配）

| 序 | 推荐 | 理由 | 前提/义务 |
| --- | --- | --- | --- |
| 1 | **foveacast-training v0.2.0 ONNX（UI 微调 MSI-Net）作为首个后端**（3s 模型为主，1s/7s 作窗口敏感性分析） | 工程：现成 ONNX + sha256/parity/quality 文档齐全，onnxruntime CPU 可跑（无 CUDA 硬需求），53.9/30.3 MB 轻量，输入输出契约精确到 dtype/值域/均值处理，describe()/predict() 封装成本最低；许可：MIT + CC-BY-4.0 署名，唯一残留为 G3/G4 上游链（内部评估不受阻）；范围匹配：静态单图、UI 域微调、1s/3s/7s 三窗口与 UEyes 基准（R2）天然对齐 | 适配层做 sum=1 重归一并声明 probability_density；limitations 记录 min-max 语义、240×320 上限、训练分布不含游戏 UI；携带 Jiang et al. 2023 与 Kroner et al. 2020 引用 |
| 2 | **DeepGaze IIE 作为通用域对照基线（仅限内部研究评估，不打包不分发）** | log_density 原生匹配契约 §3；显式 centerbias 输入匹配 §5 先验记录要求；与推荐 1 对照可**量化"UI 微调的增量价值"**（同图同窗口同指标比较，服务 G2 验收"比较可信"） | G1 裁决前仅内部评估；backend_profile 许可核查状态字段如实标"未闭合"；MIT1003 观看条件（35 px/dva、长边 1024）按 README 处理并记录假设 |
| 3 | **UEyes 数据集（CC-BY-4.0）作为公开评估基准数据**（供 R2 协议落地） | 许可闭合、体量与校验和明确（12.9 GB / md5 已记录）、自带 train/test 划分与 1s/3s/7s 真值，且与推荐 1 的训练分布同源可解释 | 下载属 >1GB 级，按 orchestration-plan §7 先报一级总控资源估计；署名义务；**游戏 UI 有效性仍需另行证明** |
| 4 | DeepGaze MSDB：暂缓，列为后续域适配观察项 | `add_dataset()` 13 标量适配通道对游戏 UI 域是低成本路线（工程线索），但 10 尺度前向计算重、pixel_per_dva 条件强、许可同 G1 未闭合 | 若启动：先过 G1 裁决 + 固定 ppd 实验配置并标记假设 |
| 5 | SeekUI：本轮不选；独立实验线存档 | scanpath 语义与第一版静态热图范围不兼容；许可三层未闭合（G7）；运行栈重（CUDA + vllm + flash-attn） | 若立项：一级总控裁决非商用边界 + 作者许可澄清 + 独立评估协议 |
| 6 | VLM 解释/候选 AOI：作为辅助线纳入设计（非后端） | UIGaze 一手证据划定能力边界（中等一致、偏长窗口探索模式）；候选 AOI 与 review 辅助价值明确且契约已预留 source=agent/candidate 通道 | 严守"不当真值"红线；具体 grounding 模型许可待补核；云端 API 须过截图上传授权关 |
| — | UMSI++：不推荐为后端 | 许可未闭合（G6）+ TF1.14/CUDA9 老栈 + notebook 形态；其论文指标可作文献参照（作者报告值） | — |

**推荐组合一句话**：以 foveacast ONNX（UI 微调）为主后端、DeepGaze IIE（仅内部）为通用域对照、UEyes test split 为公开基准数据，VLM 只做候选 AOI 与解释辅助；SeekUI/UMSI++ 存档不选。该组合同时满足"许可可辩护、工程可落地、与静态热图范围严格对齐"三条约束，且所有作者报告数字在通过 R2 协议本机复现前均不作为项目结果引用。

---

## 8. 未验证项清单（含原因）

| 项 | 原因 |
| --- | --- |
| SALICON 数据条款一手文本 | 官网不可达（DNS 失败/抓取失败），仅有二手检索来源 |
| 原始 VGG16 权重（Oxford）条款 | 一手页面未核查（时间盒） |
| MIT1003/CAT2000/Daemons/Figrim/COCO-FreeView 条款 | 时间盒收口，未逐一核查 |
| UMSI++ model_weights.zip 体量与许可、UMSI++ 骨干细节 | 按边界不下载；notebook 正文未逐行核实（时间盒收口） |
| SeekUI 论文指标、VSGUI Drive 目录条款、Qwen2.5-VL-72B 许可 | ACM DL 正文未拉取；Drive 条款页未核查；72B 仅确认被用于生成 explanation 数据 |
| UIGaze 代码/预测仓库地址 | arXiv 摘要页未列链接 |
| GUI grounding 模型（UGround/OS-Atlas/SeeClick 等）许可 | 两次定向检索超时，时间盒收口 |
| CLIP RN50x64 / DINOv2 ViT-B/14 / IIE 各骨干权重确切体量 | 按边界不下载，仅参数量推断 |
| 全部候选的本机运行可行性、推理时延、显存/内存占用、指标复现 | 本任务禁止安装与推理；留待 R3 环境实测与后续 C2 实现线 |

## 9. 与既有文档的一致性说明

- 本文档证实并细化了 `docs/sources-and-decisions.md` 中 DeepGaze 相关四条记录的现状（MIT 字段仍被注释、Issue #15 仍 open 且新增"0 评论、维护者未回应"事实）。
- D004（"DeepGaze 仅为候选"）维持有效，并新增 G1 编号缺口；本文件不修改任何既有方案文档（R1 独占范围仅本文件）。
- 建议 L2 汇总时将 §6.3 的裁决请求项转呈一级总控；将 §7 推荐 3 的 UEyes 下载体量（12.9 GB）纳入 R3/一级总控资源审批流。
