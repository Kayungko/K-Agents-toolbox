# game-ui-attention Skill 合法示例（真实运行样例）

本目录是 `skill/game-ui-attention/SKILL.md` 的配套示例：**用代码生成的合成图**完整走一遍
doctor → analyze → 读取 analysis.json → Agent 语义评审（review.json/review.md）→ summarize → compare。
文中**所有数字均来自 2026-09-11 在本项目环境下的真实执行**（退出码如实记录），不含任何虚构结果。

合法性与脱敏声明：

- 示例图片为纯 PIL/numpy 代码生成的几何抽象界面（`make_synthetic_images.py`），**不含真实游戏截图或内部资产**；"文字"以色块抽象表示。
- 信封摘录中本机绝对路径已省略/相对化；无会话 ID、无凭据。
- 运行产物在 `runs/` 下（被仓库根 `.gitignore` 的 `**/runs/` 覆盖，不入 Git，保留在本机供复核）；可提交的交付副本（请求 JSON、review.json、review.md、本 README）在 examples 顶层。
- 模型输出是**预测注意力分布**，游戏 UI 预测有效性未验证——本示例演示的是流程与评审规范，不是模型准确性证明。

## 0. 环境（doctor 真实前置检查）

```powershell
# 项目根（projects/game-ui-attention/）下执行；$py 约定见 cli-guide.md §1
& $py -m ui_attention.cli doctor
```

**退出码 0**。信封摘录（本机绝对路径与许可长文已省略）：

```json
{
  "ok": true,
  "result": {
    "tool_version": "0.1.0",
    "dependencies": {"python": "3.12.10", "numpy": "2.5.3", "pillow": "12.3.0",
                      "scipy": "1.18.1", "onnxruntime": "1.30.0"},
    "profiles": ["foveacast-onnx-3s-v1"],
    "doctor": {"ok": true, "profile_name": "foveacast-onnx-3s-v1",
                "checks": ["dependencies: ok", "device: ok (cpu, gpu_enabled=false)",
                            "weights: ok (foveacast-v3-3s-fp16.onnx, sha256=842a23f9…ef76e, 56549242 B)",
                            "license: ok (gaps G3/G4/G8, cleared_for=internal-eval)"],
                "error_codes": []}
  }
}
```

（checks 数组此处压缩为摘要行；原始信封为结构化对象，含权重 sha256 全值与 CC-BY-4.0 双署名。）

## 1. 生成合成图

```powershell
& $py skill\examples\make_synthetic_images.py
```

**退出码 0**，真实输出：

```text
synthetic-reward-summary.png: 960x540 sha256=9ef0ee388c3283bf889af7a1ace1f405c42dc36711c7b1cfd091afb8b84a89ca
synthetic-reward-summary-v2.png: 960x540 sha256=376e09d5039c17df988d9336fb7d49a4b132b3b367636bf86a4d8e3c65766add
```

图 A = 抽象"奖励结算"基线；图 B = 变体（领取按钮放大提亮、背景/标题光晕与装饰光效带减弱），同画布 960×540。

## 2. analyze（图 A 基线）

```powershell
& $py -m ui_attention.cli analyze --request skill\examples\request-analyze-a.json --out skill\examples\runs\a1-analyze-baseline
```

**退出码 0**。信封摘录：

```json
{
  "ok": true,
  "result": {
    "run_directory": "skill\\examples\\runs\\a1-analyze-baseline",
    "analysis_id": "analysis-20260911T115312Z-659682f5",
    "computation_status": "complete",
    "image_sha256": "9ef0ee388c3283bf889af7a1ace1f405c42dc36711c7b1cfd091afb8b84a89ca",
    "backend": {"backend_id": "foveacast-onnx-3s", "version": "0.1.0", "profile": "foveacast-onnx-3s-v1"},
    "artifacts": [
      {"path": "density.npy", "sha256": "0a1ee61b1aaf637421494b614a05818e18965369e57928e98154cd0f1bf8f929", "size_bytes": 4147328},
      {"path": "regions.json", "sha256": "229897e5…c328c37c", "size_bytes": 1987},
      {"path": "manifest.json", "sha256": "bc47ca5b…66a98c57", "size_bytes": 2022}
    ],
    "regions": 6,
    "limitations": [
      "原始输出为逐图 min-max 相对量，跨图绝对强度不可比",
      "输入分辨率上限 240×320，小元素可能无法解析",
      "训练分布为 2020-2022 西方语言桌面/移动 UI，不含游戏 UI，效果未验证",
      "作者公布指标未经本项目复现",
      "候选边界区域（指标同样标为基于候选边界）：['title-halo']"
    ]
  }
}
```

请求要点（`request-analyze-a.json`）：玩家目标『先查看奖励内容，再找到领取入口完成领取』；6 个区域，其中 `title-halo` 演示 Agent 自动区域规则——`source=agent`、`status=candidate`，CLI 自动把"基于候选边界"写入 limitations（见上方信封原文）。摘录说明：regions.json/manifest.json 的 sha256 以省略号截短，density.npy 保留全值供 §4 未重推理核对；原始信封为全值。

### analysis.json 关键数值（a1，真实记录；展示舍入 4-6 位）

运行环境：960×540 原图 → 320×240 推理；cpu / fp16；elapsed_ms=285.553；peak_mem_mb=300.93。

| 区域 id | role | source/status | area_fraction | probability_mass | relative_density |
| --- | --- | --- | --- | --- | --- |
| title | title | manual/confirmed | 0.020370 | 0.236654 | 11.6176 |
| reward-row | reward-content | manual/confirmed | 0.059799 | 0.260230 | 4.3517 |
| deco-band | decoration | manual/confirmed | 0.018519 | 0.032565 | 1.7585 |
| claim-button | primary-action | manual/confirmed | 0.023148 | 0.047729 | 2.0619 |
| close-button | close | manual/confirmed | 0.001975 | 0.004108 | 2.0798 |
| title-halo | decoration | **agent/candidate** | 0.081019 | 0.532025 | 6.5667 |
| region_union（掩码并集去重，overlap_dedup_px=10752） | — | — | 0.184090 | 0.875758 | 4.7572 |

## 3. Agent 语义评审（真实数值 → review.json / review.md）

- 评审文件：本目录 [review.json](review.json) / [review.md](review.md)（与运行目录 `runs/a1-analyze-baseline/` 内同名文件逐字节一致）。
- review.json **已通过冻结契约校验器真实校验**：`python -c "from ui_attention.contracts.review import load_review; load_review(...)"` → 校验通过，**退出码 0**（5 条 findings，evidence_type 分布：computed×4、unverified×1，全部携带 region_ids/evidence_refs——f5 为全图性声明，region_ids 为空列表）。
- review.md 由 review.json 呈现（四段式：观察证据→目标关系→影响推断→最小建议），未添加未记录数值。
- 评审写入位置：运行目录内（manifest 之后追加，manifest 只覆盖 CLI 计算产物；数据契约 §5 将 review.json/review.md 列为可选 Agent 产物）。

## 4. summarize（改标注不重推理）

模拟人工修正领取按钮标注（外扩含描边与外发光：380,380,200×60 → 370,370,220×80），regions 文件携带图 A 的 `image_sha256`（`regions-summarize-a2.json`）：

```powershell
& $py -m ui_attention.cli summarize --analysis skill\examples\runs\a1-analyze-baseline --regions skill\examples\regions-summarize-a2.json --out skill\examples\runs\a2-summarize-regions
```

**退出码 0**。信封摘录：

```json
{
  "ok": true,
  "result": {
    "analysis_id": "analysis-20260911T115728Z-cde2abef",
    "source_analysis_id": "analysis-20260911T115312Z-659682f5",
    "density_sha256": "0a1ee61b1aaf637421494b614a05818e18965369e57928e98154cd0f1bf8f929",
    "recomputed_regions": 6
  }
}
```

- **未重推理证据**：`density_sha256` 与 §2 analyze 信封 `artifacts` 中 density.npy 的 sha256（0a1ee61b…f8f929）完全一致。
- 标注外扩后 claim-button 真实新值：area_px=17600、area_fraction=0.033951、mass=0.064813、relative_density=1.9091——mass 变大主要来自面积扩大，相对密度反而低于原标注的 2.0619（面积≠效率的又一实例）。
- a2 运行目录复用了源目录的 review.json 并重渲染报告（CLI 行为）。**注意**：区域数值已变，该复用评审对 claim-button 的数值已过期——真实流程中 Agent 必须按新 analysis.json 重写评审，不沿用过期结论（SKILL.md 步骤 6）。

## 5. analyze（图 B 变体）+ compare（A/B）

```powershell
& $py -m ui_attention.cli analyze --request skill\examples\request-analyze-b.json --out skill\examples\runs\b1-analyze-variant
& $py -m ui_attention.cli compare --before skill\examples\runs\a1-analyze-baseline --after skill\examples\runs\b1-analyze-variant --out skill\examples\runs\c1-compare-a-b
```

两步均**退出码 0**。b1 analysis_id=`analysis-20260911T115729Z-aba9f30f`（elapsed_ms=287.801）。compare 信封摘录：

```json
{
  "ok": true,
  "result": {
    "comparison_schema": "game-ui-attention-comparison/v1",
    "compatible": true,
    "exploratory": false,
    "matched_regions": 6,
    "added_regions": [],
    "removed_regions": [],
    "limitations": [
      "delta_pp 只说明模型预测分布发生变化，不表示点击率或任务效率因而提高（validation-plan §5）",
      "新增/移除区域未编造缺失一侧的 0 值，不参与差值计算"
    ]
  }
}
```

### comparison.json 真实 delta_pp（稳定 ID 配对，6/6）

| 区域 id | mass A→B | delta_pp | area_px A→B | relative_density A→B |
| --- | --- | --- | --- | --- |
| claim-button | 0.047729 → 0.052350 | **+0.4621** | 12000 → 17280（+5280） | 2.0619 → **1.5705 ↓** |
| close-button | 0.004108 → 0.005091 | +0.0983 | 1024 → 1024 | 2.0798 → 2.5774 |
| deco-band | 0.032565 → 0.027659 | -0.4907 | 9600 → 9600 | 1.7585 → 1.4936 |
| reward-row | 0.260230 → 0.251951 | -0.8278 | 31000 → 31000 | 4.3517 → 4.2133 |
| title | 0.236654 → 0.250573 | +1.3919 | 10560 → 10560 | 11.6176 → 12.3008 |
| title-halo（candidate） | 0.532025 → 0.559426 | +2.7401 | 42000 → 42000 | 6.5667 → 6.9049 |

### 这次真实运行示范的三条红线

1. **面积变化与效率提升不得混同**：claim-button 的 mass 上升 +0.46pp，但其面积同时从 12000 扩到 17280 px，relative_density 反而从 2.06 降到 1.57——按红线口径，这次改动**不能**表述为"按钮视觉效率提升"，只能说"预测质量增加，且增加与面积扩大同时发生，单位面积吸引力在预测分布中下降"。
2. **预测方向不保证符合设计改动直觉，如实报告不美化**：图 B 有意减弱了标题光晕与装饰，但 title-halo（候选边界）与 title 的 mass 在预测分布中反而上升。本示例不为该现象编造因果解释；正确处置是如实报告数值、保留候选边界声明、必要时做进一步敏感性分析。
3. **共用色阶由 CLI 保证**：c1 manifest 记录 A/B 联合值域色阶 `ember vmin=7.38228e-11 vmax=4.53888e-05`（覆盖两侧：A 单独 vmax=4.32499e-05、B 单独 vmax=4.53888e-05），双方 overlay 同一标尺渲染——Agent 不需要也不允许各自归一化后用"更红=更好"下结论。

## 6. 文件清单

| 文件 | 性质 |
| --- | --- |
| `make_synthetic_images.py` | 合成图生成脚本（真实执行，exit 0） |
| `synthetic-reward-summary.png` / `-v2.png` | 合成图 A/B（960×540，可公开） |
| `request-analyze-a.json` / `request-analyze-b.json` | AnalyzeRequest v1（真实消费） |
| `regions-summarize-a2.json` | summarize 用修改标注（携带 image_sha256，真实消费） |
| `review.json` / `review.md` | Agent 评审副本（与 a1 运行目录内一致；契约校验 exit 0） |
| `runs/a1-analyze-baseline/` | analyze 真实产物（含 review.json/review.md） |
| `runs/a2-summarize-regions/` | summarize 真实产物 |
| `runs/b1-analyze-variant/` | 图 B analyze 真实产物 |
| `runs/c1-compare-a-b/` | compare 真实产物（comparison.json + A/B 报告） |

## 7. 复核方式（L2 复跑口径）

```powershell
# 项目根下：重新生成图（sha256 应与 §1 一致，脚本含固定随机种子）→ 用新输出目录名重跑 §2/§4/§5 命令
# 注意：--out 已存在会按契约返回退出码 7（不覆盖历史），复跑请换新目录名
```
