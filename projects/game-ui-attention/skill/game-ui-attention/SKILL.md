---
name: game-ui-attention
description: 驱动 ui-attention CLI 对游戏 UI 截图做视觉注意力评估：截图 → 预测热图 → AOI 区域统计 → 可追溯语义评审，支持改标注不重推理（summarize）与同配置 A/B 对比（compare）。当用户要求分析游戏 UI 截图的视觉注意力、生成注意力热图、统计区域（AOI）占比/密度，或对界面修改前后做 A/B 对比评审时使用。模型输出是预测分布，不是真实眼动数据。
---

# game-ui-attention：游戏 UI 视觉注意力评估

## 触发条件

用户提出以下任一诉求时启用本 Skill：

- 分析一张游戏 UI 截图的视觉注意力 / 显著性 / 热图；
- 统计指定区域（奖励、按钮、标题、立绘等 AOI）在预测注意力分布中的占比或相对密度；
- 评估"关键内容是否突出、装饰是否竞争注意力"，或按玩家目标做视觉层级评审；
- 对同一界面的修改前后（A/B）做注意力分布对比；
- 修改 AOI 标注后重算区域统计（不重新推理）。

不适用（第一版范围外，直接说明而不要勉强分析）：眼动采集、视频/时序注视轨迹、点击率预测、HUD 动态内容、自动改图或写回 Unity/Figma。

## 角色分工（先读）

- **程序负责数值，Agent 负责语义解释。** 一切概率、占比、密度、差值只能来自 CLI 产物（`analysis.json` / `comparison.json`），Agent 不计算、不心算、不虚构任何数值。
- **玩家目标影响评审优先级，不改变基础热图。** 不给任何指定区域人为加权。
- Skill 侧只做三件事：组织输入（请求 JSON / AOI 标注）、按退出码调用 CLI、基于产物数值形成可追溯评审。

## 红线（最高优先级，任何一条冲突时立即停止并向用户说明）

1. **模型输出是预测注意力分布，不是真实眼动。** 不得表述为"玩家实际看了/会看哪里"，不得输出真实注视时长、首眼顺序、注视次数。
2. **禁止编造数值类结论**：注视时长、首眼顺序、点击率、模型置信百分比、通用 UI 质量分数（如"注意力评分 87/100"）一律不得出现在任何输出中。第一版不设"主按钮必须达到某百分比"的合格线。
3. **占比语义红线**：`probability_mass = 0.08` 只能说"预测分布中约 8% 的概率质量落在该区域"，**不得**说"8% 的玩家会看到"或"注视时间占 8%"。
4. **面积与效率不得混同**：区域尺寸变化引起的 mass 增长必须同时报告 `area_fraction` 变化，不得把面积扩大带来的占比增长解释成"视觉效率提升"；判断效率看 `relative_density`（mass/面积占比）并结合区域角色。
5. **后端失败禁止造假**：CLI 非零退出时，不得自行绘制热图、不得用 VLM 目测猜测冒充数值、不得改用其他云端服务出图。正确动作：如实报告退出码与 `error.id`；`MODEL_NOT_READY` → 建议运行显式安装入口（见调用指南）；其余 → 报告阻塞，等待修复。
6. **游戏 UI 预测有效性未验证**：本工具未在眼动数据上验证对游戏 UI 的预测准确性（训练分布不含游戏 UI）。每份评审必须如实声明这一点，不得暗示"已验证有效"。
7. **VLM 不当真值**：视觉语言模型（含宿主自身的看图能力）只能产出候选 AOI（`source=agent`、`status=candidate`）与语义观察；VLM 的观察在评审中只能标 `observed` / `inferred` / `unverified`，**不得标 `computed`**，不得当眼动真值、评估答案或热图来源。
8. **A/B 色阶红线**：A/B 共用颜色标尺由 CLI 保证，Agent 不得各自归一化两张热图，不得用"看起来更红/更亮"作为比较结论——比较只读 `comparison.json` 的浮点数值与 `delta_pp`。
9. **截图与隐私**：仅分析用户主动提供或明确授权分析的截图；本地推理不自动上传截图；若宿主 Agent 需要调用云端视觉模型解释图片，必须事先向用户披露该事实（不能宣称端到端离线）；未获授权的截图不得发往任何外部 API。
10. **不覆盖历史**：运行产物写入新目录；输出目录已存在（退出码 7）时换新目录名，不得删除或覆盖旧结果。

## 调用约定（摘要）

完整命令、退出码分支表与错误处置见同目录 [cli-guide.md](cli-guide.md)。核心口径：

- CLI 未 pip install，经 venv + `PYTHONPATH` 调用：`$env:PYTHONPATH = "<项目根>\src"`，解释器 `<项目根>\.venv\Scripts\python.exe -m ui_attention.cli <command> ...`（Windows PowerShell）。
- stdout 是**一个 JSON 信封**（`{"ok": true, "result": {...}}` 或 `{"ok": false, "error": {id, message, details, exit_code}}`），进度信息在 stderr。**按退出码分支，不做文字匹配。**
- 权重缺失时先运行显式安装入口 `<项目根>\.venv\Scripts\python.exe -m ui_attention.backends`（带 sha256 校验；绝不静默替换模型）。
- 运行产物写入 `runs/` 下**新目录**（`--out` 指定；已存在 → 退出码 7）。`runs/`、截图与权重均不入 Git。
- 当前登记 profile：`foveacast-onnx-3s-v1`（请求中的 `backend_profile` 只能用已登记名，未登记 → 退出码 3）。

## 标准流程

### 步骤 1：doctor 前置检查

```powershell
& $py -m ui_attention.cli doctor
```

- 退出码 0：读取信封中 `result.profiles`（确认请求将使用的 profile 已登记）、`result.doctor.checks`（依赖/设备/权重/许可）。
- 退出码 3（`MODEL_NOT_READY`）：向用户报告并建议运行安装入口；用户同意后再执行安装，**不自动下载**。
- doctor 不通过时不进入 analyze。许可记录含缺口（如 G3/G4/G8）且 `cleared_for=internal-eval` 时，产出仅限内部评估用途，向用户如实转述。

### 步骤 2：收集输入，构造 AnalyzeRequest

向用户确认三件事：**截图文件路径**、**玩家目标**、**关注的区域**。

- 截图缺少文件路径时，使用宿主真实提供的附件机制获取，**不猜测路径**。
- 玩家目标（`player_goal`）可缺省；缺省时只做描述性评审，或在评审中显式注明所采用的目标假设。缺目标时不得声称"目标完成路径合理"。
- `screen_type` 按界面类型填写（如 `reward-summary`、`shop`、`equipment-detail`、`quest`、`popup`），供评审组织参考，自由字符串。

AOI 标注规则（构造 `regions` 时逐条遵守）：

| 规则 | 要求 |
| --- | --- |
| 坐标系 | 原图像素坐标，**原点为左上角**，基于方向处理后的原图 |
| 矩形 | 半开边界 `[x, x+width) × [y, y+height)`；宽高必须为正（零面积拒绝） |
| 多边形 | `points: [[x,y], ...]`，至少三个不共线顶点，不得自交 |
| 边界 | 坐标必须在原图内（越界拒绝，退出码 2） |
| ID | 同一请求内唯一，A/B 对比时保持稳定（配对靠 ID） |
| source | `manual`（人工标注）\| `agent`（Agent/VLM 自动识别）\| `imported`（设计节点导入，必须附 `source_reference` 与实际坐标变换记录，不得仅凭节点名称关联截图） |
| status | `candidate` \| `confirmed`。**Agent 自动识别的区域必须 `status=candidate`**；`confirmed` 仅表示边界经人工确认，不表示效果通过验证 |

Agent 自动圈定候选区域（含借助 VLM 识图）时：区域标 `source=agent`、`status=candidate`，报告中展示边框，**相关指标与结论必须同样标注"基于候选边界"**。人工确认后才可升级 `confirmed`。

没有可靠区域信息时允许全图分析（`regions` 缺省），不虚构精确的按钮占比。嵌套/重叠区域允许，但指标不可简单求和——汇总看 CLI 产出的 `region_union`（掩码并集去重）。

请求文件示例（v1 schema，相对图片路径相对**请求文件所在目录**解析）：

```json
{
  "schema_version": "game-ui-attention-request/v1",
  "image": "./screenshot.png",
  "player_goal": "查看奖励内容并找到领取入口",
  "screen_type": "reward-summary",
  "backend_profile": "foveacast-onnx-3s-v1",
  "regions": [
    {
      "id": "claim-button",
      "label": "领取按钮",
      "role": "primary-action",
      "geometry": {"type": "rect", "x": 380, "y": 420, "width": 200, "height": 64},
      "source": "manual",
      "status": "confirmed"
    }
  ]
}
```

`role` 为自由字符串，建议采用场景规则中的角色词表（见 [scenario-rules.md](scenario-rules.md)）。

### 步骤 3：analyze

```powershell
& $py -m ui_attention.cli analyze --request <request.json> --out runs\<新目录名>
```

- 退出码 0：从信封记录 `analysis_id`、`image_sha256`、`limitations`，然后进入步骤 4。
- 非零：按 [cli-guide.md](cli-guide.md) 的退出码分支表处置；失败不产生部分产物（CLI 自清理），不得基于失败运行继续评审。
- 透明通道图片默认拒绝；仅在用户给出真实合成背景时用 `--composite-background "R,G,B"`，不擅自填黑或白。

### 步骤 4：读取 analysis.json 数值

评审只允许消费运行目录内 `analysis.json` 的记录值：

- `regions[]`：每区域的 `area_px`、`area_fraction`、`probability_mass`、`relative_density`、`source`、`status`、`geometry`；
- `region_union`：重叠去重后的汇总（含 `overlap_dedup_px`）；
- `limitations`：后端已知限制（相对量语义、240×320 分辨率上限、训练分布不含游戏 UI）、候选区域提示、缺目标提示——**评审必须转述相关项**；
- `model` / `profile` / `runtime`：实际后端、权重哈希、配置哈希、设备与耗时（回答"用什么算的"时引用）；
- `input`：图片哈希与原图/推理尺寸。

展示热图用 `overlay.png`（叠加图）与 `report.html`（本地打开，可圈选导出标注）；**统计只读浮点数值，不从 PNG 颜色反推任何指标**。

### 步骤 5：Agent 语义评审（review.json + review.md）

按 technical-design §8 的四段式组织每条发现：**观察证据 → 目标关系 → 影响推断 → 最小建议**。

每条 finding 必须携带：

| 字段 | 要求 |
| --- | --- |
| `id` | 非空且唯一（如 `f1-claim-button-mass`） |
| `region_ids` | 涉及的 AOI id 列表（全图性发现可为空列表，但优先落到具体区域） |
| `evidence_refs` | 证据引用列表，指向产物内可核查的位置（如 `analysis.json#/regions/claim-button/probability_mass`、`overlay.png`）；不得放空想数值 |
| `evidence_type` | 四分类之一，见下表 |
| `observation` | 观察证据：只陈述产物记录值与画面可见事实 |
| `inference` | 先写与玩家目标的关系，再写影响推断；推断不得越过证据边界 |
| `recommendation` | 最小建议：单一、可执行、不改范围的下一步（含"需要补充验证什么"时写明） |
| `validation_needed` | 字符串（需要什么验证）或布尔 |

`evidence_type` 判定（每条 finding 取其主要证据性质）：

| 类型 | 含义 | 典型例子 |
| --- | --- | --- |
| `computed` | 程序计算的产物数值 | probability_mass、relative_density、delta_pp |
| `observed` | 画面上可见的事实（含 VLM 观察） | "按钮为金色高对比底色"、"弹窗遮罩压暗背景" |
| `inferred` | 由证据推出的影响判断 | "领取入口与奖励内容竞争同一视觉通道" |
| `unverified` | 缺少验证、只能存疑的陈述 | "该层级问题是否影响实际领取率未验证" |

禁止事项：不提供未经校准的模型置信百分比；不从单张截图得出"用户一定看不到""点击率会提升""真实注视顺序"结论；HUD 运动、场景切换、玩家熟练度属未覆盖条件，涉及即标 `unverified`。

`review.json` 结构（与 `contracts/review.py` 冻结校验一致，**不得添加未知字段**）：

```json
{
  "schema_version": "game-ui-attention-review/v1",
  "analysis_id": "<analysis.json 中的 analysis_id>",
  "created_at_utc": "2026-09-11T00:00:00Z",
  "author": "<agent 标识>",
  "findings": [
    {
      "id": "f1",
      "region_ids": ["claim-button"],
      "evidence_refs": ["analysis.json#/regions/claim-button/probability_mass"],
      "evidence_type": "computed",
      "observation": "……（只引用记录值）",
      "inference": "……（目标关系 + 影响推断）",
      "recommendation": "……（最小建议）",
      "validation_needed": "……"
    }
  ]
}
```

写入位置：运行目录内 `review.json` 与 `review.md`（数据契约 §5 的可选 Agent 产物；manifest 在其之前已落盘，表示计算完成——计算完成不代表评审完成）。`review.md` 由 `review.json` 内容呈现，**不能额外添加未记录的数值**。评审失败时保留有效计算结果，如实报告"计算完成、评审未完成"。

### 步骤 6（可选）：summarize / compare

- **summarize**（改标注不重推理）：用户在 `report.html` 圈选导出、或 Agent 修改标注后，构造独立 regions 文件（`schema_version=game-ui-attention-regions/v1`，**必须携带对应图片的 `image_sha256`**，防止标注落到其他图片），运行 `summarize --analysis <旧运行目录> --regions <regions.json> --out <新目录>`。产物 `density_sha256` 与源一致即证明未重推理。评审结论随新区域数值更新，`review.json` 若存在于源目录会被复用重渲染——区域数值变了就要重写评审，不沿用过期结论。
- **compare**（A/B）：两次 analyze 的运行目录满足同模型、同 profile、同预处理、同画布尺寸时，运行 `compare --before <run-A> --after <run-B> --out <新目录>`，读取 `comparison.json` 的 AOI 配对表与 `delta_pp`。退出码 6（`COMPARISON_INCOMPATIBLE`）时不强行比较、不换工具凑数，向用户报告不兼容原因。

## A/B 对比规则

1. 对比双方必须：同一模型与权重、同一 `backend_profile`（同预处理）、同一画布尺寸；中心偏置与观看配置一致（由 CLI 兼容性校验把关，Agent 不绕过）。
2. AOI 通过**稳定 ID** 配对；位置和大小可以改变，但两侧各自提供有效边界。新增/删除的区域单独列出（`added`/`removed`），**不编造缺失一侧的 0 值**；不匹配区域不计算差值。
3. 共用色阶由 CLI 写入双方 manifest；Agent 不得各自归一化，不得用"更红=更好"这类颜色语言下结论，比较结论只引用 `delta_pp` 与两侧记录值。
4. `delta_pp` 解读同样受红线约束：只说明"模型预测分布发生了变化"，**不得**声称"点击率或任务效率因此提高"。
5. 画面状态、任务或内容明显不同（不同关卡、不同弹窗状态）时，标记为**探索性比较**（信封 `exploratory` 字段），不作修改的因果结论。
6. 区域尺寸变化时，`delta_pp` 必须与两侧 `area_fraction` 一起呈现，区分"面积扩大"与"单位面积吸引力变化"（看 `relative_density`）。

## 场景规则

奖励结算 / 商店 / 装备详情 / 任务页 / 弹窗五类界面的区域角色词表与预期层级建议见 [scenario-rules.md](scenario-rules.md)。它们是**设计经验规则，非验证结论**：只用于组织评审优先级与 role 命名，不构成数值阈值，不改变热图，不作为"合格/不合格"判据。

## 输出与呈现要求

- 每份评审开头声明：结果基于模型预测分布（后端与 profile 实名）、游戏 UI 预测有效性未验证；有候选区域时声明"基于候选边界"。
- 引用数值时给到区域 id 与字段名，保持与 `analysis.json` 记录一致；展示舍入不得改变原始记录值。
- 产物（`report.html`、`overlay.png`）明确标注"模型预测"语义；运行目录、截图不入 Git。
- 用户要求"能不能证明玩家会这样看"时，按验证方案口径回答：独立眼动评估之前，只能声明"模型预测分布如何"，不能声明真实玩家行为。

## 合法示例

`skill/examples/` 提供端到端真实运行样例（合成图，非真实游戏截图）：生成脚本、请求 JSON、真实 CLI 信封摘录、基于真实数值的 `review.json` / `review.md`，以及 summarize/compare 的真实运行证据。见 [examples/README.md](../examples/README.md)。
