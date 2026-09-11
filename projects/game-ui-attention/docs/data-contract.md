# 数据与接口草案

状态：设计草案，尚无 CLI 或 schema 实现。示例全部为虚构输入，不是分析结果。

## 1. 拟定命令

```text
ui-attention doctor
ui-attention analyze --request request.json --out <new-run-directory>
ui-attention summarize --analysis <run-directory> --regions regions.json --out <new-run-directory>
ui-attention compare --before <run-A> --after <run-B> --out <new-run-directory>
```

`doctor` 检查依赖、模型文件与哈希、设备能力及许可记录，不打印凭据；下载模型是独立的显式安装步骤。`analyze` 完成计算与基础报告，Agent 根据其结果生成语义评审。`summarize` 复用已有概率图、只更新区域指标。

## 2. AnalyzeRequest v1

```json
{
  "schema_version": "game-ui-attention-request/v1",
  "image": "./input.png",
  "player_goal": "查看奖励并领取",
  "screen_type": "reward-summary",
  "backend_profile": "local-static-v1",
  "regions": [
    {
      "id": "claim-button",
      "label": "领取按钮",
      "role": "primary-action",
      "geometry": {"type": "rect", "x": 800, "y": 900, "width": 320, "height": 96},
      "source": "manual",
      "status": "confirmed"
    }
  ]
}
```

- 相对图片路径相对于 request 文件目录解析，不相对于 shell 当前目录。
- `player_goal` 和 `regions` 可缺省；缺少目标时不得声称目标完成路径合理。
- `backend_profile` 必须是已登记、不可在执行时静默改写的配置；示例 `local-static-v1` 尚不存在。
- 像素坐标基于处理方向后的原图。矩形范围 `[x, x + width) × [y, y + height)`；多边形采用 `points: [[x,y], ...]`，至少三个不共线顶点。
- 坐标必须在原图内；拒绝重复 ID、零面积、多边形自交和未知 schema 版本。
- 标注来源为 `manual | agent | imported`，状态为 `candidate | confirmed`。`confirmed` 表示边界经确认，不表示模型或设计效果通过验证。
- 导入设计节点时附加 source reference 和实际坐标变换记录；不得仅凭节点名称关联截图。
- 修改标注的独立 regions 文件必须携带对应图片 SHA-256，防止应用到其他图片。

## 3. 后端接口（G0 冻结签名，2026-09-11）

本节为 C1（contracts/ 定义）与 C2（backends/ 实现）之间的冻结接口；任何签名变更必须递增 `schema_version` 并经二级总控重新冻结，实现线不得单方面改动。

```text
describe() -> BackendInfo
predict(image, resolved_profile) -> PredictionResult
```

约定（Python 类型口径，实现于 `contracts/backend.py`）：

- `predict(image, resolved_profile)` 的 `image` 为方向已处理的原图 `numpy.ndarray`，RGB、`uint8`、形状 `(H, W, 3)`；缩放、均值减除等模型侧预处理由后端按 `resolved_profile` 内部执行并写入 `shape_mapping`，调用方不得预处理（foveacast 的均值减除在图内完成，见技术方案 §3）。
- `resolved_profile` 为登记 profile 的解析结果（见下），含预处理参数、可选 centerbias 引用、观看条件与全部配置哈希；后端拒绝未登记或运行时改写的 profile（退出码 3）。

`BackendInfo` 字段（冻结）：

| 字段 | 类型/取值 | 说明 |
| --- | --- | --- |
| backend_id | str | 如 `foveacast-onnx-3s`、`deepgaze-iie` |
| version | str | 后端实现版本（独立于模型权重版本） |
| capabilities | tuple[str] | `spatial_density` / `vendor_metrics` / `scanpath`；第一版只消费 `spatial_density` |
| native_semantics | str | 原生输出语义：`log_density` \| `probability_density` \| `vendor_metrics` |
| device_requirements | dict | `{"device": "cpu"\|"cuda", "min_free_vram_mb": int\|None}`；doctor 实测空闲显存后校验 |
| license_status | dict | `{"code": "...", "weights": "...", "gaps": ["G1", ...], "cleared_for": "internal-eval"\|"packaged"}`；缺口编号沿用 sources-and-decisions |
| weights | tuple[WeightRef] | 每项 `{name, source_url, sha256, size_bytes}`；doctor 校验实际文件哈希==登记值，不匹配退出码 3（MODEL_NOT_READY） |

`PredictionResult` 字段（冻结）：

| 字段 | 类型/取值 | 说明 |
| --- | --- | --- |
| semantics | str | 显式声明 `log_density` \| `probability_density` \| `vendor_metrics`；foveacast 适配层完成 sum=1 重归一化后声明 `probability_density`，DeepGaze IIE 原生 `log_density` |
| array | numpy.ndarray | float64；空间图形状 `(h, w)`；不得为 8-bit 或量化存储 |
| shape_mapping | dict | 原图尺寸 `(H, W)`、推理尺寸 `(h, w)`、缩放/填充参数与逆变换方法（对齐技术方案 §4.4-4.5） |
| runtime | dict | `{"device", "precision", "elapsed_ms", "peak_mem_mb"}` 实测值，写入 analysis.json 的 runtime 字段 |
| vendor_metrics | dict \| None | 仅 `vendor_metrics` 语义时保存原名称与语义，不强制转换成 probability_mass |
| limitations | tuple[str] | 必须包含：相对量语义（如 min-max 跨图不可比）、分辨率上限（如 240×320）、训练分布边界（不含游戏 UI）等后端已知限制 |

时序能力（scanpath）不属于第一版消费范围；SeekUI 类后端若未来立项，须以独立 backend_profile 与独立评估协议接入（D013）。

**profile 登记（冻结规则）**：`backend_profile` 必须是 `backends/registry.py` 中已登记、不可在执行时静默改写的配置。登记名规范 `<backend>-<variant>-v<N>`（示例请求中的 `local-static-v1` 为虚构占位，首个真实登记预期为 `foveacast-onnx-3s-v1`）。登记项包含：backend_id+version、权重清单与哈希、预处理参数（目标尺寸、值域、均值处理）、centerbias 引用（可选，含来源与哈希）、观看条件假设（如 DeepGaze 的 35 px/dva 须标记实验假设）、全部配置的合成哈希。分析记录必须包含实际执行的后端配置解析结果，不能只保存请求中的别名。

## 4. 计算结果与证据

| 字段 | 内容 |
| --- | --- |
| schema_version | `game-ui-attention-analysis/v1` |
| analysis_id | 每次运行生成的唯一 ID |
| computation_status | `complete | failed` |
| review_status | `not_requested | pending | complete | failed` |
| evidence_type | `model_prediction` |
| input | 图片哈希、原图及推理尺寸、画面类型 |
| model | 后端 ID、版本、全部权重哈希、代码版本与 capability |
| profile | 预处理、中心偏置及观看条件，含假设来源 |
| runtime | 依赖版本、设备类型、精度、耗时与资源实测 |
| regions | 每个 AOI 的边界、来源、状态、面积、mass、relative_density |
| limitations | 候选区域、未知观看条件、动态内容未覆盖等 |
| artifacts | 相对路径及哈希；缺少的产物不列为成功 |
| errors | 结构化错误，不带凭据或原始第三方敏感响应 |

语义评审单独保存为 `review.json`：每条 finding 包含 `id`、`region_ids`、`evidence_refs`、`evidence_type`、`observation`、`inference`、`recommendation` 和 `validation_needed`。`review.md` 由该内容呈现，不能额外添加未记录的数值。

计算完成不代表评审完成，也不代表模型预测准确。

## 5. 产物

```text
<run-directory>/
  manifest.json       产物与完整性记录
  analysis.json       数值与运行证据
  regions.json        区域边界和来源
  density.npy        全精度空间概率图（支持该能力时）
  base.png           原图展示副本（G1 补录：报告自包含用；哈希绑定仍以原始输入文件 sha256 为准）
  overlay.png        展示用热图
  report.html        查看、圈选区域、导出标注
  review.json        可选的 Agent 评审
  review.md          可选的评审文本
```

报告尽量自包含；嵌入原图的 HTML 也属于截图数据，不自动纳入公开仓库。HTML 中的区域名称、玩家目标和评审文本必须转义，外部脚本和自动网络请求默认关闭。

## 6. 比较结果

`comparison.json` 保存左右 analysis ID、配置兼容性、画面可比性说明、AOI 匹配表、原始值与 `delta_pp`。新增/移除区域单独记录；不匹配区域不计算差值。共享色阶参数写入 manifest。

预处理、模型、先验、画布尺寸等不兼容时拒绝正式比较，保留原因；不能依靠“都叫 attention”跨厂商直接比较。

## 7. 拟定退出码

| 退出码 | 意义 |
| --- | --- |
| 0 | 请求的计算步骤成功 |
| 1 | 内部错误（未预期工具缺陷；保证任何失败均非零退出，不产生假成功。G0 后按 C1 阶段报 A 偏差申报补录，2026-09-11） |
| 2 | 参数、图片或 AOI 无效 |
| 3 | 后端、依赖、权重或许可配置未就绪 |
| 4 | 设备资源不足或推理失败 |
| 5 | 输出概率无效或产物校验失败 |
| 6 | A/B 配置不兼容 |
| 7 | 输出路径已存在、不可写或文件操作失败 |

stdout 输出一个 JSON 信封，包含 `ok`、`result` 或 `error`；进度发到 stderr。不以文字匹配代替退出码分支。示例错误 ID：`MODEL_NOT_READY`、`INVALID_AOI`、`GPU_OOM`、`INVALID_DENSITY`、`COMPARISON_INCOMPATIBLE`。
