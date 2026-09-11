# Game UI Attention

游戏 UI 视觉注意力评估项目。拟定 Skill 名称：`game-ui-attention`；拟定 CLI 名称：`ui-attention`。

**状态：G0-G3 四阶段全部验收（2026-09-11/12）——CLI+Skill 交付、535 项测试通过、UEyes 公开基准评估完成（验收记录：[G1](docs/acceptance/g1-engineering-loop.md)/[G2](docs/acceptance/g2-comparison.md)/[G3](docs/acceptance/g3-tool-delivery.md)）。核心实测：主候选 foveacast-3s 在原生窗口显著优于中心偏置基线（IG_CB +0.552，胜率 92.6%），UI 微调较通用域 DeepGaze 增量约 1.06 bits；S2 全窗口硬门槛口径修订待一级总控裁决。尚无安装包或发布物；游戏 UI 预测有效性未验证（无眼动数据）。**

## 目标

输入完整的游戏界面截图、可选的玩家目标和重点区域标注，生成可追溯的预测热图、区域指标和视觉层级评审；支持修改前后对比。

例如：输入“奖励结算界面，希望玩家先看奖励，再找到领取按钮”，检查奖励、领取按钮与装饰内容的预测分布及其设计含义。

## 第一版范围

- 静态完整截图分析。
- 重点区域（AOI）标注和统计。
- 原图与热图叠加展示。
- 相同分析配置下的 A/B 对比。
- 由 Agent 结合玩家目标解读计算结果。
- 本地 HTML 报告和结构化 JSON 结果。

第一版不实现眼动采集、视频时序模型、点击预测、Unity/Figma 自动写入、自动改图或训练新模型。空间热图不输出“真实注视时长”“首眼顺序”“玩家看到的概率”或统一 UI 质量分数。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [技术方案](docs/technical-design.md) | 架构、选型（G0 锁定）、算法、隐私与失败行为 |
| [数据与接口草案](docs/data-contract.md) | CLI、输入输出、后端冻结签名、状态和比较条件 |
| [实施计划](docs/implementation-plan.md) | 阶段、交付物、C1/C2/C3 目录所有权和完成标准 |
| [验证方案](docs/validation-plan.md) | 工程正确性、公开数据评估协议引用与证据边界 |
| [来源与选型记录](docs/sources-and-decisions.md) | 公开来源、许可缺口（G1-G8）与设计决策（D001-D016） |
| [分级执行计划](docs/orchestration-plan.md) | 一级/二级总控职责、子任务分工和阶段验收 |
| [研究波次状态](docs/research/README.md) | 第一波研究索引与验收状态（模型候选/评估协议/环境可行性） |
| [G1 验收记录](docs/acceptance/g1-engineering-loop.md) | 工程闭环验收：范围、实测检查、失败案例、未验证项与结论 |
| [G3 验收记录](docs/acceptance/g3-tool-delivery.md) | 工具交付验收：AOI 重算、A/B 色阶、模型哈希、Skill 封装 |
| [G2 验收记录](docs/acceptance/g2-comparison.md) | 公开数据比较验收：UEyes 实跑、S0-S4 判定、泄漏审计与双划分对照、缺陷披露链 |
| [Skill](skill/game-ui-attention/SKILL.md) | 原创 game-ui-attention Skill（红线、流程、调用指南与真实运行示例） |

## 当前进展

- [x] 定义最小范围与技术路线。
- [x] 整理接口、实施和验证草案。
- [x] 明确模型代码、权重及依赖的使用许可（G0/R1：首选后端 foveacast ONNX 许可闭合；DeepGaze/SeekUI/UMSI++ 缺口 G1/G6/G7 记录在案待裁决；商用链残留 G3/G4/G8 如实标注）。
- [x] 核实本机运行条件并选择可用后端（G0/R3 实测 + R1 短名单：路线 A foveacast ONNX CPU 先行；后端跑通验证属阶段 0 剩余工作）。
- [x] 锁定公开数据评估协议（G0/R2：UEyes + 七指标统一口径；评估未运行；成功判据待一级总控确认）。
- [x] 实现并测试工程管线（G1 验收通过：CLI 四命令、契约/AOI/概率统计/评估脚本/报告渲染三线交付，475 项测试通过，真实后端全链跑通）。
- [x] 封装实际 Skill（G3 验收通过：skill/game-ui-attention/SKILL.md+调用指南+场景规则+真实运行示例，见 [docs/acceptance/g3-tool-delivery.md](docs/acceptance/g3-tool-delivery.md)；宿主运行时行为验证属使用侧）。
- [x] 公开数据基准评估（G2 验收通过：UEyes test 108 图×3 窗口×4 后端+基线，表 A-E+bootstrap CI+泄漏审计+备选划分对照；S2 按批准口径判定，口径修订待一级总控裁决；详见 [G2 记录](docs/acceptance/g2-comparison.md)）。
- [ ] 评估模型在游戏 UI 上的适用性（无眼动数据前保持未验证；公开 UI 结果不得外推为游戏结论）。

文档创建日期：2026-09-11。来源核查日期见来源记录与各研究文档（均为 2026-09-11）；硬件环境已实测（R3），模型运行兼容性未实测。
