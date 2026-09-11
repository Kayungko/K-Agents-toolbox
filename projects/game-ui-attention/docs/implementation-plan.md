# 实施计划

状态：G0 研究收敛完成（2026-09-11，见 [research/](research/README.md)）；工程实现（阶段 0 后半 + 阶段 1-3）即将按 C1/C2/C3 三线派发。以下阶段均未通过验收。

## 阶段 0：模型与运行条件验证

工作：

- ~~核实候选模型代码、权重、骨干模型及依赖的使用和分发许可~~ → **G0 已完成**（R1：首选后端 foveacast ONNX 许可闭合；DeepGaze/SeekUI/UMSI++ 缺口编号 G1/G6/G7 待裁决；商用链残留 G3/G4/G8 记录在案）。
- ~~确认 Python、依赖、GPU/CPU 条件~~ → **G0 已完成实测与设计**（R3：venv py3.12、缓存重定向、依赖 pin 与 lock 方案、体量分级）；创建隔离环境实体归 C2 执行，不改全局 Python。
- 使用获准的公开或合成图片验证输入输出、权重加载和概率语义（C2：foveacast 3s FP16 下载→sha256 校验→加载→合成图推理→min-max 输出重归一化为 probability_density；消解 U6/U8）。
- 测冷启动、热推理、峰值显存/内存，记录输入尺寸和设备（C2：回填 analysis runtime 字段；消解 U4/U5）。
- 固定初始预处理、中心偏置和配置版本（C2 提案、经 registry 登记冻结为 `foveacast-onnx-3s-v1`）。

交付：后端选型记录（**G0 已交付**：model-candidates.md + sources-and-decisions D011-D016）、依赖锁定文件（C2 建 venv 后由 C1 合入 pyproject/lock）、运行测量表（C2）。许可不明确的后端不进入正式交付（DeepGaze 仅内部评估对照、待 G1 裁决），不阻止模型无关的统计与报告模块。

完成标准：至少一个可按明确许可使用的后端输出有效密度（路线 A foveacast，<100MB 无需下载审批）；记录实测而非宣传值。尚未要求证明游戏 UI 预测准确。

## 阶段 1-3 实现线目录所有权（G0 锁定，布局全文见 runtime-feasibility.md §3）

三线最多并行，各自只写自己标注的目录；跨线只经由 contracts/ 冻结接口协作；共享文件唯一持有者为 C1：

| 线 | 独占目录/文件 | 职责 |
| --- | --- | --- |
| C1 | `src/ui_attention/{__init__,cli,errors,imaging}.py`、`contracts/`、`aoi/`、`metrics/`（含 eval/ 评估脚本）、`pyproject.toml`、lock 文件、`tests/fixtures/`、`tests/unit/test_{contracts,imaging,aoi,metrics}_*.py` | 契约 schema 与校验、四命令入口与退出码、图片预处理、AOI、概率统计、R2 协议评估脚本；C2/C3 的依赖项以提案→C1 合入 |
| C2 | `src/ui_attention/backends/`、`.venv/`（忽略）、`model-cache/`（忽略）、`tests/unit/test_backends_*.py` | venv 与缓存重定向、registry/doctor/weights、foveacast ONNX 适配层（及获批后的 DeepGaze 内部对照）、runtime 实测 |
| C3 | `src/ui_attention/report/`、`tests/unit/test_report_*.py` | overlay.png、A/B 共用色阶、report.html（自包含、转义、禁外链）、区域圈选导出、comparison 视图 |
| L2（总控） | `tests/integration/`、跨模块整合、README、全部 git 提交 | 串行整合与 G1 闭环验收 |

## 阶段 1：单图最小闭环

工作：实现 doctor、analyze、后端接口、输入校验、概率转换、产物 manifest 和基本报告。先使用手动 AOI。

交付：完整单图计算链、结构化错误、可追溯产物、数值和坐标检查。

完成标准：同一图片与配置能重复计算；无效输入和推理失败不会生成假成功；报告数值与 JSON 一致。

## 阶段 2：AOI 编辑与 A/B

工作：实现矩形/多边形标注、JSON 导出、summarize、稳定 ID 配对、兼容性检查及共用色阶。

交付：可编辑区域的本地报告、区域重算、comparison.json 和对比报告。

完成标准：改 AOI 不重新推理；重叠汇总去重；不同配置拒绝比较；面积变化和新增/删除区域正确呈现。

## 阶段 3：Agent Skill

工作：编写原创 `game-ui-attention` Skill，调用已验证 CLI，区分观察、计算与推断，按界面用途解释结果。

交付：SKILL.md、最少必要的场景规则、调用指南和合法示例。

完成标准：Agent 能完成截图到报告流程；不会在后端失败时自行画热图；不会编造注视时长、置信度和设计质量分数。若截图缺少文件路径，使用宿主真实提供的附件机制，不猜测路径。

## 阶段 4：游戏 UI 试用

建议先准备 20～30 张有使用许可的截图，覆盖结算、商店、装备、任务、弹窗和 HUD，包含不同背景复杂度和主次层级。

交付：适用场景、失败案例、延迟和建议质量记录。该规模只是工程起点，不代表统计检验能力。

完成标准：按[验证方案](validation-plan.md)分别报告工程正确性、评审有用性和预测有效性；没有眼动数据时预测有效性保持未验证。

## 后续候选

- 批量推理和模型缓存服务。
- Figma 区域信息导入与 Unity 运行截图接入（新增需求时另定范围）。
- 获得眼动数据后的模型评估及适配。
- 视频、动态 HUD 与任务条件建模。
- 多宿主共用 CLI/MCP。

## 阶段推进原则

每阶段记录实际变更、检查结果、尚未覆盖条件与下一步。不能将规划文档、合成输入或代码结构检查写成“模型准确”“游戏验证通过”或“可正式发布”。
