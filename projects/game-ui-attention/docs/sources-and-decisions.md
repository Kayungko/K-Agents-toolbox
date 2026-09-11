# 来源与选型记录

文档日期：2026-09-11。G0 接口锁定更新：2026-09-11（并入第一波研究 R1/R2/R3 结论，详见 [docs/research/](research/README.md) 三份文档；本文件保留原始记录并标注修订）。下列事实来自官方仓库、厂商文档和论文页面的在线核查；本项目尚未进行模型安装或推理测试。正式选型时固定 commit 和资产哈希（R1 已为各候选记录 pin commit）。

## 1. 公开来源

### 1.1 初始方案阶段来源（2026-09-11 上午）

| 来源 | 已读取信息 | 对本方案的影响 |
| --- | --- | --- |
| [DeepGaze 官方仓库](https://github.com/matthias-k/DeepGaze) | IIE 空间 log density、显式中心偏置；MSDB 要求 pixels per degree；III 使用注视历史 | 首版考虑静态密度，不从空间图推断时间；先验和观看配置需记录 |
| [DeepGaze setup.py](https://github.com/matthias-k/DeepGaze/blob/main/setup.py) | torch/torchvision 等依赖；MIT 分类与 license 字段被注释 | 不能据此宣称项目已按 MIT 授权 |
| [DeepGaze IIE 实现](https://github.com/matthias-k/DeepGaze/blob/main/deepgaze_pytorch/deepgaze2e.py) | 多骨干组合、预训练权重加载 | 需要完整依赖与权重清单；不得将其视为无需测量的轻量模型 |
| [DeepGaze 许可 Issue #15](https://github.com/matthias-k/DeepGaze/issues/15) | 商业使用许可问题仍为打开状态 | 正式集成与分发前明确许可；不将"公开"写成"可自由商用" |
| [Attention Insight 功能](https://attentioninsight.com/features/) | 热图、AOI 占比、焦点图与清晰度功能 | 可作为产品能力参考；不直接复制其专有分数语义 |
| [Attention Insight API](https://attentioninsight.com/api/) | 可将设计评估接入产品 | 外部后端候选 |
| [Attention Insight MCP](https://app.attentioninsight.com/mcp/instruction) | 官方远程 MCP、账户认证和分析额度 | Agent 集成候选；不是已接入的能力 |
| [Neurons 热图说明](https://knowledge.neuronsinc.com/neurons-ai-heatmaps-and-interpretation) | 多种预测图与 AOI 含义 | 不混淆注意力、复杂度和其他厂商指标 |
| [Neurons 插件说明](https://knowledge.neuronsinc.com/neurons-ai-plugins) | Figma 和浏览器集成 | 后续交互流程参考 |
| [UEyes：CHI 2023](https://doi.org/10.1145/3544548.3581096) | 不同 UI 类型上的视觉显著性研究与模型比较 | 通用图片模型的表现不能直接推广到游戏 UI |

厂商的准确率宣传未作为本项目指标或承诺。外部链接引用用于说明来源，不代表已获得其实现、数据或模型的分发许可。

### 1.2 第一波研究新增核实来源（2026-09-11，R1/R2/R3 逐项核查，完整清单见各研究文档）

| 来源 | 关键事实 | 状态 |
| --- | --- | --- |
| [foveacast-training](https://github.com/khawkins98/foveacast-training)（pin `65209a2d`）+ [Release v0.2.0](https://github.com/khawkins98/foveacast-training/releases/tag/v0.2.0) | UI 微调 MSI-Net；代码 LICENSE=MIT（全文核实）；发布 ONNX（1s/3s/7s × FP16 53.9MB/INT8 30.3MB）逐文件带 sha256；输入 240×320 RGB [0,255]，输出逐图 min-max [0,1]；onnxruntime CPU 可跑；权重使用须携带 UEyes 署名（Jiang et al. 2023） | 已核实（R1 §C2） |
| [alexanderkroner/saliency](https://github.com/alexanderkroner/saliency)（pin `bd81e582`）+ [HF MSI-Net](https://huggingface.co/alexanderkroner/MSI-Net) | MSI-Net 上游代码 MIT（全文核实）；HF 权重卡 `license: mit`；encoder 由 VGG16 ImageNet 初始化、SALICON 训练 | 已核实（R1 §C2） |
| [UEyes 仓库](https://github.com/YueJiang-nj/UEyes-CHI2023)（pin `7bc06417`）+ [Zenodo 8010312](https://zenodo.org/records/8010312) | 数据集 CC-BY-4.0（Zenodo API 核实）；单 zip 12.9GB、md5 已录；62 人 × 1,980 UI × 4 类；1s/3s/7s 窗口为同一 trial 前缀截断（官方源码核实）；info.csv 含官方 train/test 划分；UMSI++ 代码仓无 LICENSE、TF1.14/CUDA9 老栈 | 已核实（R1 §C1、R2 §2.1） |
| [DeepGaze](https://github.com/matthias-k/DeepGaze)（pin `c7db17e2`）+ Issue #15 API 实时核查 | 仓库无 LICENSE、setup.py MIT 字段注释；Issue #15 open、0 评论、维护者未回应；Release 权重无许可声明；IIE 输出原生 log_density、显式 centerbias；MSDB 需 pixel_per_dva、10 尺度前向；骨干含 CLIP（MIT）与 DINOv2（Apache-2.0） | 已核实（R1 §C3）→ 许可缺口 G1 |
| [SeekUI-CHI2026](https://github.com/YueJiang-nj/SeekUI-CHI2026)（pin `404a3074`）+ HF checkpoint | 输出 scanpath 坐标序列（时序能力，不兼容静态热图接口）；代码 LICENSE 缺失与 MIT 徽章矛盾；基座经参数量比对确认为 Qwen2.5-VL-3B（仅非商用许可，全文核实） | 已核实（R1 §C4）→ 缺口 G7 |
| [UIGaze（arXiv 2604.26352）](https://arxiv.org/abs/2604.26352) | 9 个 SOTA VLM 在 UEyes 1,980 张全量零样本评估：与真人眼动仅中等一致，随观看窗口变长而提高 | 已核实摘要级（R1 §C5）；其代码仓库未验证 |
| [Kümmerer et al., ECCV 2018](https://arxiv.org/abs/1704.08615) + pysaliency 源码 + [MIT 基准站](http://saliency.mit.edu/) | 指标权威定义（IG/NSS/CC/KL/SIM/AUC 族）三方交叉核实；σ≈1° 视角约定；8-bit 量化警示 | 已核实（R2 §6、§10） |
| [UEyes 官方评估代码](https://github.com/YueJiang-nj/UEyes-CHI2023)（evaluation/） | 与标准口径存在系统性偏差：IG 无真实基线、KL 方向相反、AUC 用 0.7 相对阈值、NSS 用模糊图掩码；论文 Table 2 有校准勘误 | 已核实源码（R2 §6.8）→ 复现分数与论文分数禁止混表 |
| PyPI / download.pytorch.org / nvidia-smi 等环境与体量来源 | torch 2.14.0 CPU win wheel 124.1MB、onnxruntime 1.30.0（14.7MB，MIT）；本机 RTX 4070 12GB、64GB RAM、C 盘仅 60GB 可用、无系统级 CUDA Toolkit | 已实测/已核实（R3 §1、§4） |

不可达或未核实来源（FiWI 官方页许可、WebSaliency 仓库 404、SALICON 官网、saliency.tuebingen.ai、UEyes 论文 PDF 正文）均已在研究文档中标注"未验证+原因"。

## 2. 拟定决策

| ID | 决策 | 理由 |
| --- | --- | --- |
| D001 | 独立项目，首版 CLI + Skill | 可以先验证计算与评审，再按需要接宿主 |
| D002 | 静态完整画面为首版范围 | 减少时序和任务条件的不确定性 |
| D003 | 模型计算与语义评审分离 | 数值可追溯；目标不会人为篡改热图 |
| D004 | DeepGaze 仅为候选（**G0 修订：许可缺口 G1 确认未闭合——无 LICENSE、Issue #15 open 0 评论；限内部研究评估对照，不打包不分发；使用边界待一级总控裁决**） | 许可与实际运行、游戏效果尚未闭合 |
| D005 | 输出预测概率占比、面积与相对密度 | 避免大面积区域占比高被直接判为问题 |
| D006 | 不使用统一 UI 质量分数 | 缺少跨场景校准依据 |
| D007 | 原始浮点概率图为统计来源 | 不从伪彩图反推数值 |
| D008 | A/B 固定配置及共用色阶 | 避免模型或展示配置变化造成假差异 |
| D009 | 后端失败明确返回错误 | 不静默降级、替换模型或上传截图 |
| D010 | 接入设计工具放到后续 | 首先完成最小分析闭环 |
| D011 | **首选后端候选：foveacast-training v0.2.0 ONNX（UI 微调 MSI-Net，3s FP16 主用、1s/7s 窗口敏感性）**（G0 新增） | 许可闭合（代码 MIT+权重署名传递，内部评估无阻碍；商用前需关闭 G3/G4/G8）；工程摩擦最低（onnxruntime CPU、53.9MB、sha256 齐备、输入输出契约精确）；UI 域微调与静态热图范围严格对齐。适配层须做 sum=1 重归一并声明 probability_density；limitations 记录 min-max 相对量语义、240×320 上限、训练分布不含游戏 UI |
| D012 | **公开评估基准：UEyes 数据集（CC-BY-4.0）+ R2 协议口径**（G0 新增） | 许可闭合、md5/体量明确、自带官方划分与三窗口真值；评估协议见 [benchmark-protocol.md](research/benchmark-protocol.md)；12.9GB 下载待一级总控批准，获批前评估代码用合成数据自测 |
| D013 | **SeekUI 不进入第一版；scanpath 搜索线单独存档**（G0 新增） | 输出语义与静态热图接口不兼容（时序能力）；许可三层未闭合（G7）；运行栈重。若立项需一级总控裁决非商用边界 |
| D014 | **UMSI++ 不选为后端**（G0 新增） | 代码无 LICENSE（G6）+ TF1.14/CUDA9 老栈 + notebook 形态；论文指标仅作文献参照（作者报告值） |
| D015 | **VLM 仅作候选 AOI 与解释辅助，不是显著性后端**（G0 新增） | UIGaze 一手证据：零样本 VLM 与真人眼动仅中等一致；红线：VLM 输出不得当眼动真值、评估答案或热图来源；候选区域走 `source=agent, status=candidate` 通道 |
| D016 | **运行环境：venv py3.12 + 缓存全部重定向项目内忽略目录 + 路线 A（ONNX CPU）先行**（G0 新增） | R3 实测：C 盘仅 60GB 可用而 D 盘 1TB；无系统级 CUDA Toolkit → onnxruntime-gpu 首版不选；DeepGaze 老 API × 新 torch 兼容性风险（U1）开工前隔离验证 |

## 3. 尚待确定（G0 更新）

已闭合（第一波研究完成）：

- ~~正式后端及其代码、权重、骨干与依赖许可~~ → 首选后端许可闭合（D011）；DeepGaze/SeekUI/UMSI++ 缺口已编号（G1/G6/G7）。
- ~~可用硬件、锁定依赖、可接受输入尺寸与运行成本~~ → 硬件实测完成、依赖 pin 与体量分级完成（R3）；foveacast 输入 240×320 已确认，游戏 UI 高分辨率截图的缩放策略在 C1 imaging 模块实现并按契约记录。
- ~~是否有独立眼动数据~~ → 确认无自采数据；公开数据评估协议已锁定（R2）。

待一级总控裁决/批准（已随 G0 汇总上报）：

- G1：DeepGaze 仅内部评估对照的使用边界。
- G3：SALICON 条款一手不可达——foveacast/MSI-Net 权重链商用性处置（本轮限内部使用，商用前补核）。
- UEyes 12.9GB 下载授权（G2 验收关键路径）。
- R2 协议七项待裁决（成功判据 S0-S4、真值加权、IG 基线、sAUC 口径、FiWI 纳入、重复阈值/排除门槛冻结值），见 benchmark-protocol.md §11。
- G8：ImageNet 类残留风险的发布披露口径（发布前确定即可）。

仍待后续阶段确定：

- 仓库原创内容的许可证（本次不代替维护者选择分发条款）。
- 初始中心偏置配置的正式版本（评估用 CB 按 R2 §4.2 由 train 划分构造；分析用先验按 technical-design §5 版本化登记）。
- 有使用许可的游戏 UI 试用样本与标注（阶段 4）。

## 4. 更新原则

模型被替换、配置改变或获得新实测证据时，更新此记录和项目状态，并使旧报告仍能追溯其原始版本。运行成功、人工认可和真实眼动验证分别记录。许可缺口编号（G1-G8）与兼容性未知项编号（U1-U18）沿用研究文档口径，关闭时在对应研究文档与本记录同步标注。
