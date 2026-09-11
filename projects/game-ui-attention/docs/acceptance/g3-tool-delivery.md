# G3 工具交付验收记录

验收人：二级总控（L2）。验收日期：2026-09-11。模板依据：[validation-plan.md](../validation-plan.md) §6。

## 版本与范围

- 提交基线：G1 记录（`5ef1bf3`）之后的 `skill/` 交付（S1 线，本记录随附提交）。
- 范围：G3 阶段四项——AOI 重算、A/B 共用色阶、真实模型与依赖哈希、原创 Skill 封装及使用说明。**不含** G2（公开数据比较可信——待 UEyes 下载批准）与游戏 UI 有效性（无眼动数据）。

## 实际检查与结果（全部真实执行，L2 独立取证）

| # | G3 要求 | 证据 | 结果 |
| --- | --- | --- | --- |
| 1 | AOI 重算（改标注不重推理） | L2 集成测试 test_summarize_no_reinference：summarize 后 density.npy sha256 与 analyze 完全一致、区域统计随新标注更新（进程级，真实 CLI） | 通过 |
| 2 | A/B 共用色阶 | L2 集成测试 test_compare_chain：compare 双方 manifest 记录同一 colorscale 参数；冒烟证据 delta_pp 实值（-2.345pp/0.0）；C3 单测含"共用参数两侧一致、A 侧不被单独拉满"断言 | 通过 |
| 3 | 真实模型与依赖哈希 | 权重 sha256 L2 独立计算=release digest（842a23f9…585ef76e，56,549,242B）；PROVENANCE.json 溯源；requirements.txt lock 与 venv 实装一致（C1 复核）；doctor 输出权重哈希校验+许可记录 | 通过 |
| 4 | 原创 Skill 封装及使用说明 | S1 交付 SKILL.md（10 条红线+六步流程+finding 八字段/evidence_type 四分类+A/B 六规则+截图隐私规则）、cli-guide.md（命令/退出码分支/错误处置）、scenario-rules.md（五类界面角色词表，grep 核验无数值阈值）、examples/（端到端真实运行样例） | 通过 |
| 5 | Skill 示例真实性（L2 复跑） | 用 skill/examples/request-analyze-a.json 独立复跑 analyze：exit 0、六区域数值与 examples/README.md 记录**逐字一致**（claim-button relative_density=2.0619 精确匹配，确定性推理佐证）；README 含红线应用示范（面积扩大 mass +0.46pp 但 relative_density 2.06→1.57，正确表述为"不得称视觉效率提升"） | 通过 |
| 6 | 越范围检查 | git status 仅 skill/ 新增；examples/runs/ 被 **/runs/ 规则忽略（check-ignore 实证），重产物不入仓 | 通过 |
| 7 | Skill 行为红线（文档级验收） | 后端失败不自绘热图/不 VLM 冒充（红线 5）、不编造注视时长与置信度（红线 2）、游戏 UI 有效性未验证必声明（红线 6）、占比语义（红线 3）、历史不覆盖（红线 10）均成文且与冻结契约一致 | 通过 |

## 失败案例记录

- 无。S1 全程增量落盘（前波 C3 截断教训已内置派发词），无越范围写入，无虚假数值。

## 未验证项

- **G2（比较可信）整体待一级总控批准 UEyes 12.9GB 下载**：评估脚本已就绪（S0 合成自检通过），固定测试划分/中心基线/逐图指标须在真实数据上运行后才可宣称。
- Skill 在真实宿主 Agent 会话中的行为验收（Agent 实际按 SKILL.md 执行不违规）：文档级验收通过，运行时行为属宿主侧使用验证。
- HTML 圈选交互的浏览器 E2E：静态断言通过（同 G1 记录）。
- 1s/7s 窗口敏感性权重：未下载（评估开工前由 L2 批准）。

## 结论

**G3 工具交付：通过**（AOI 重算、A/B 共用色阶、真实模型与依赖哈希、原创 Skill 及使用说明四项证据闭合）。G0/G1/G3 已闭合；**G2 为当前唯一待外部输入的阶段**（UEyes 下载批准 + S0-S4 判据确认）。产物均标注模型预测语义与许可门禁；游戏 UI 预测有效性维持"未验证"口径不变。
