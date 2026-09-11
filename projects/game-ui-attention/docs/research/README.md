# 研究波次状态与所有权（脱敏）

本目录存放分级执行计划第一波研究产出。此文件由二级总控（L2）维护，只使用任务代号；会话 ID、派发回执与机器路径保存在仓库外协调记录中，不写入本仓库。

状态口径：`进行中` = 已派发未验收；`待验收` = 执行者已回报；`已验收` = L2 按验收标准核对通过。

## 第一波（研究，2026-09-11 派发，同日全部验收通过）

| 代号 | 任务 | 独占产物 | 状态 | 验收要点 |
| --- | --- | --- | --- | --- |
| R1 | 模型候选与许可核实（UMSI++/UEyes、UI 微调 MSI-Net/foveacast、DeepGaze IIE/MSDB、SeekUI、VLM 解释路线） | docs/research/model-candidates.md | **已验收**（L2 逐条核验通过，已提交） | 五候选均有来源+核查日期+许可四件套结论；缺口 G1-G8 单列；作者报告与本机复现严格区分；推荐短名单：foveacast ONNX 首选后端、DeepGaze IIE 仅内部对照、UEyes 作基准数据、SeekUI/UMSI++ 存档不选、VLM 仅辅助 |
| R2 | 公开数据评估协议（划分、基线、窗口、IG/NSS/CC/sAUC/KL 口径、反泄漏） | docs/research/benchmark-protocol.md | **已验收**（L2 逐条核验通过，已提交） | 指标公式含数值防护可直接实现；反泄漏五规则+可比性八项清单成文；与 UEyes 官方实现的口径差异表防混表；成功判据 S0-S4 标注"待一级总控确认"；无自采数据边界显著声明 |
| R3 | 运行环境实测与隔离方案（venv/缓存/依赖、最小 CLI 布局、下载体量估计） | docs/research/runtime-feasibility.md | **已验收**（L2 逐条核验通过，已提交） | 附录 A 实测证据齐全且脱敏；venv(py3.12)/缓存重定向/依赖 pin 方案可直接照做；CLI 布局与契约逐项映射且 C1/C2/C3 目录所有权不重叠；体量分级表 >1GB 项标注"需一级总控批准"；兼容性未知项 U1-U18 成节 |

约束（执行期间均遵守）：R1/R2/R3 未修改既有五份方案文档、项目配置或其他执行者文件；未执行任何 git 写操作；产物由 L2 显式逐文件暂存、集中本地提交。

## L2 保留所有权

- 既有方案文档（technical-design / data-contract / implementation-plan / validation-plan / sources-and-decisions）的接口锁定更新：待第一波收敛后由 L2 串行处理。
- README 导航与真实状态更新、.gitignore 追加、全部本地提交。

## 第二波（实现，2026-09-11 派发）

接口已锁定（G0 提交）；共享 venv 基础设施已由 L2 预建（Python 3.12.10，numpy/pillow/scipy/onnxruntime/pytest/ruff 实装，缓存重定向到忽略目录）。三线并行、目录所有权不重叠（全文见 [implementation-plan.md](../implementation-plan.md) 所有权表）：

| 代号 | 范围 | 独占产物 | 状态 |
| --- | --- | --- | --- |
| C1 | 契约 schema、AOI、概率统计、错误语义、评估脚本；共享文件（pyproject/cli.py/errors.py/fixtures）唯一持有者 | src/ui_attention/{contracts,aoi,metrics}/、cli.py、errors.py、imaging.py、pyproject.toml、tests/fixtures/ | **已验收**（276 单测+接线核验，G1 记录） |
| C2 | 后端隔离环境维护、foveacast ONNX 下载校验/注册/doctor/推理适配、运行实测 | src/ui_attention/backends/、model-cache/（忽略） | **已验收**（82 单测+sha256 独立复算一致+实测数字，G1 记录） |
| C3 | 本地报告、区域导出、A/B 共用色阶、HTML 安全渲染 | src/ui_attention/report/ | **已验收**（110 单测+HTML 安全抽查，G1 记录） |
| S1 | 原创 game-ui-attention Skill 封装 | skill/ | **已验收**（红线全覆盖+示例复跑数值逐字一致，G3 记录） |

验收记录：[G1 工程闭环](../acceptance/g1-engineering-loop.md)、[G2 比较可信](../acceptance/g2-comparison.md)、[G3 工具交付](../acceptance/g3-tool-delivery.md)。G2 已于 2026-09-12 以 UEyes 真实数据执行完毕（一级总控批准下载+判据；S2 口径修订仍待裁决，判定按批准原文口径记录）。

许可门禁不变：DeepGaze 线待一级总控 G1 裁决（C2 禁止安装 torch）；UEyes 12.9GB 下载待批准（C1 eval 以合成数据自测）；跨线整合与 tests/integration/ 由 L2 串行处理。
