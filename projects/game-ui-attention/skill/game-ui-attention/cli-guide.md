# ui-attention CLI 调用指南（Windows PowerShell 口径）

本指南是 `game-ui-attention` Skill 的配套文件：完整命令示例、stdout 信封格式、退出码分支表与常见错误处置。口径与 `docs/data-contract.md`（冻结契约）及 `src/ui_attention/errors.py` 一致；冲突时以冻结契约为准。

## 1. 前置条件

- CLI **未 pip install**，通过项目 venv + `PYTHONPATH` 调用（不发明新调用方式）。
- 环境：`<项目根>\.venv`（Python 3.12），依赖 numpy / pillow / scipy / onnxruntime 已装。
- 权重：缓存于 `<项目根>\model-cache\foveacast\`（不入 Git）。**权重缺失时的唯一正确动作是运行显式安装入口**（见 §2），带 sha256 校验；绝不静默替换模型或改用其他来源。

约定下文 `$py` 指：

```powershell
# 在 <项目根>（projects/game-ui-attention/）下执行
$env:PYTHONPATH = "$PWD\src"
$py = "$PWD\.venv\Scripts\python.exe"
```

## 2. 命令总览（四命令 + 安装入口）

```powershell
# 0) 权重显式安装入口（仅权重缺失时；成功/已缓存 exit 0，结构化失败 exit 3）
& $py -m ui_attention.backends

# 1) 前置检查：依赖、模型文件与哈希、设备能力、许可记录（不打印凭据）
& $py -m ui_attention.cli doctor

# 2) 分析：截图 → 概率热图 → AOI 统计 → 运行产物（--out 必须是新目录）
& $py -m ui_attention.cli analyze --request <request.json> --out runs\<新目录名>
#    透明图片仅在用户给定真实合成背景时：追加 --composite-background "R,G,B"（0-255）

# 3) 改标注重算（复用既有概率图，不重新推理）
& $py -m ui_attention.cli summarize --analysis runs\<既有运行目录> --regions <regions.json> --out runs\<新目录名>

# 4) A/B 对比（兼容性校验 + 稳定 ID 配对 + delta_pp）
& $py -m ui_attention.cli compare --before runs\<run-A> --after runs\<run-B> --out runs\<新目录名>
```

要点：

- `analyze --request`：请求内相对图片路径**相对请求文件所在目录**解析，不相对 shell 当前目录。
- `--out`：运行目录必须**不存在**（不覆盖历史）；已存在 → 退出码 7。常规放 `runs/` 下（git-ignored）。
- `summarize --regions`：独立 regions 文件必须携带对应图片的 `image_sha256`（`schema_version=game-ui-attention-regions/v1`），防止标注应用到其他图片。
- `compare`：两侧画布尺寸、模型、profile（预处理/先验）必须一致，否则退出码 6。

## 3. stdout 信封与 stderr

- **stdout 只输出一个 JSON 信封**；进度信息全部在 stderr。解析 stdout 即可，不要混流。
- 成功：`{"ok": true, "result": {...}}`
- 失败：`{"ok": false, "error": {"id": "<错误ID>", "message": "...", "details": {...}, "exit_code": <int>}}`
- **按退出码分支，不做文字匹配**；`error.id` 只用于向用户报告与选择处置动作。
- 信封与产物中可能包含本机绝对路径（如权重缓存目录），对外转述时省略或替换为相对路径（脱敏）。

## 4. 退出码分支表（data-contract §7 冻结）

| 退出码 | 含义 | Agent 动作 |
| --- | --- | --- |
| 0 | 请求的计算步骤成功 | 读取 `result` 与运行产物，继续流程 |
| 1 | 内部错误（未预期工具缺陷；`INTERNAL_ERROR`） | 报告阻塞并附信封原文；不重试凑数、不绕过 CLI 手工产数 |
| 2 | 参数、图片或 AOI 无效 | 按 `error.id` 修正输入后重试（见 §5）；不得改语义迁就（如擅自裁图、删区域凑通过） |
| 3 | 后端、依赖、权重或许可配置未就绪 | `MODEL_NOT_READY` → 经用户同意运行安装入口（§2 命令 0）；`PROFILE_NOT_REGISTERED` → 改用 doctor 列出的已登记 profile 名；`LICENSE_NOT_CLEARED` → 报告许可阻塞 |
| 4 | 设备资源不足或推理失败（`GPU_OOM` / `INFERENCE_FAILED`） | **报告阻塞，不重试造假**：不得自行绘制热图或用 VLM 猜测代替数值；可向用户建议关闭占用后重试一次，仍失败即停 |
| 5 | 输出概率无效或产物校验失败（`INVALID_DENSITY` / `ARTIFACT_CHECK_FAILED` / `UNSUPPORTED_SEMANTICS`） | **报告阻塞**：这是工具/产物完整性问题，禁止修补产物或凭空补数 |
| 6 | A/B 配置不兼容（`COMPARISON_INCOMPATIBLE`） | 向用户转述不兼容原因（`details`）；不强行比较、不跨配置凑对比结论 |
| 7 | 输出路径已存在、不可写或文件操作失败（`OUTPUT_PATH_EXISTS` / `IO_ERROR`） | `OUTPUT_PATH_EXISTS` → **换一个新输出目录名重跑**，不删除不覆盖历史；`IO_ERROR` → 检查路径权限后报告 |

失败路径不产生假成功：非零退出时 CLI 自清理本次目录，无 manifest；**不得基于失败运行继续评审**。

## 5. 常见错误处置速查

| error.id（退出码） | 典型原因 | 处置 |
| --- | --- | --- |
| `MODEL_NOT_READY` (3) | 权重缺失 / sha256 不符 / 后端模块缺失 | 报告用户 → 同意后运行 `& $py -m ui_attention.backends`（显式安装、哈希校验）→ 重跑 doctor → 再 analyze。哈希不符时不得跳过校验强行使用 |
| `PROFILE_NOT_REGISTERED` (3) | `backend_profile` 用了未登记名（如虚构的 `local-static-v1`） | 从 doctor 信封 `result.profiles` 取已登记名（当前为 `foveacast-onnx-3s-v1`）；不发明 profile |
| `INVALID_AOI` (2) | 零面积 / 越界 / 重复 ID / 多边形自交或共线 / 枚举值非法 | **修标注**：对照 error.details 逐条修正区域几何或 id/source/status 后重跑；不通过删除区域掩盖问题（除非用户同意） |
| `INVALID_IMAGE` (2) | 图片不可解码 / 带透明通道且未给合成背景 | 请用户提供完整已合成 RGB 截图；确有透明且用户给定背景色时用 `--composite-background` |
| `INVALID_REQUEST` / `INVALID_SCHEMA_VERSION` (2) | 请求字段拼写错、未知字段、schema 版本不对 | 按契约 v1 修正请求 JSON；不猜测版本迁移 |
| `REGIONS_IMAGE_MISMATCH` (2) | summarize 的 regions 文件 `image_sha256` 与源分析图片不符 | 用源 `analysis.json` 的 `input.image_sha256` 重建 regions 文件；不得改哈希迁就 |
| `GPU_OOM` (4) | 显存/内存不足 | 报阻塞；建议释放资源后经用户同意重试一次；仍失败停止并报告，不降级造假 |
| `INFERENCE_FAILED` (4) | 推理执行失败 | 同上：报阻塞，不自行绘制热图 |
| `INVALID_DENSITY` (5) | 概率图 NaN/负值/sum≠1 超差 | 报阻塞（工具缺陷级），附信封原文 |
| `ARTIFACT_CHECK_FAILED` (5) | 产物缺失/哈希不符（如 density.npy 被改动） | 报阻塞；不手改产物；必要时经用户同意用原图重新 analyze 到新目录 |
| `COMPARISON_INCOMPATIBLE` (6) | A/B 两侧模型/权重/预处理/画布尺寸不一致 | 转述原因；只有在两侧用同一 profile、同画布尺寸重新 analyze 后才可再比 |
| `OUTPUT_PATH_EXISTS` (7) | `--out` 目录已存在 | 换带时间戳/序号的新目录名重跑；**绝不删除或覆盖历史运行** |
| `INTERNAL_ERROR` (1) | 未预期缺陷 | 附完整信封报阻塞，等待修复 |

## 6. 运行产物清单（analyze 成功后的运行目录）

```text
<run-directory>/
  manifest.json    产物与完整性记录（最后落盘 = 计算完成；含色阶/overlay 参数）
  analysis.json    数值与运行证据（评审唯一数值来源）
  regions.json     区域边界和来源（含 image_sha256）
  density.npy      全精度 float64 概率图（sum=1；统计与 summarize 的数据源）
  base.png         原图展示副本（报告自包含用；哈希绑定以 analysis.input.image_sha256 为准）
  overlay.png      展示用热图叠加（固定透明度与色阶；仅展示，不用于取数）
  heatmap.png      热图展示件
  report.html      本地报告：查看、圈选区域、导出标注（无外链、无自动网络请求）
  review.json      Agent 评审（步骤 5 写入；CLI 只校验与呈现，不生成数值）
  review.md        评审文本（由 review.json 呈现，不加未记录数值）
```

summarize 运行目录额外在 manifest 记录 `source_analysis_id` 与 `density_sha256`（与源一致 = 未重推理）；compare 运行目录产出 `comparison.json` 与 A/B 报告，共享色阶参数写入 manifest。

## 7. 验证建议（Agent 自检口径）

- analyze 成功后核对信封 `result.computation_status == "complete"` 且 `artifacts` 中列出 manifest.json；缺失产物不列为成功（报告渲染失败时计算仍有效，属"部分完成"，如实转述）。
- summarize 成功后核对 `result.density_sha256` 与源运行一致。
- compare 成功后核对 `result.compatible == true`，并记录 `exploratory` 标志与 `added/removed` 区域列表。
- 任何数值引用可回溯到 `analysis.json` / `comparison.json` 的具体字段路径。
