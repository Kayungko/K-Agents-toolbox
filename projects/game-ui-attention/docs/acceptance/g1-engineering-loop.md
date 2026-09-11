# G1 工程闭环验收记录

验收人：二级总控（L2）。验收日期：2026-09-11。模板依据：[validation-plan.md](../validation-plan.md) §6。

## 版本与范围

- 仓库提交基线：`a4ab2ac`（C1 core）/`f872c09`（C2 backends）/`f088d92`（C3 report）/`4cb5d24`（C1 tests）；本验收新增 `tests/integration/`（L2 独占）。
- 范围：阶段 0 后半（真实后端跑通+运行测量）+ 阶段 1（单图最小闭环）+ 阶段 2（AOI 编辑与 A/B）的工程正确性。**不含**预测有效性（无眼动数据）与游戏 UI 适用性评估（阶段 4）。

## 输入与配置

- 后端：`foveacast-onnx-3s-v1`（登记 profile，config_hash `94dc0ccb…0b068`）；权重 foveacast-v3-3s-fp16.onnx，56,549,242 B，sha256 `842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e`（L2 独立计算=release digest）。
- 环境：Windows 主机（RTX 4070 12GB/CPU i7-14700KF/64GB RAM，R3 脱敏口径），venv Python 3.12.10，onnxruntime 1.30.0 CPU EP，numpy 2.5.3，pillow 12.3.0，scipy 1.18.1。
- 输入图：合成渐变图（640×480 与 1920×1080），无任何真实游戏截图。

## 实际检查与结果（全部真实执行，L2 独立取证）

| # | 检查 | 结果 |
| --- | --- | --- |
| 1 | 全量单测（规范口径：项目根 cwd+配置自动发现） | **468 passed in 4.11s**（C1 276+C2 82+C3 110，三线集成零回归） |
| 2 | L2 进程级集成测试（subprocess 真实 CLI+真实后端） | **7 passed in 7.26s**；全套件合计 **475 passed in 10.72s** |
| 3 | analyze 全链（真实推理） | exit 0；产物 manifest/analysis/regions/density.npy/overlay/heatmap/report.html 齐全；manifest 全部工件 sha256+size 独立复算一致；manifest 最后落盘成立 |
| 4 | 概率语义 | density.npy float64、有限、非负、sum=1（<1e-6） |
| 5 | summarize 不重推理 | density.npy sha256 与 analyze 完全一致；区域扩大后 area_px 增大、统计更新 |
| 6 | compare 链 | exit 0；delta_pp 真实数值（改动区域 -2.35pp、未动区域 0.0）；A/B 共用色阶参数写入双方 manifest（同 name/参数一致） |
| 7 | 退出码分支（进程级） | 0（成功）/2（零面积 AOI→INVALID_AOI，无部分产物残留）/3（未登记 profile）/7（输出目录已存在，历史 manifest 字节级未被覆盖）；1/4/5/6 由 C1 单测覆盖 |
| 8 | report.html 安全抽查 | 0 个 http(s) 外链引用、0 个 `<script src>`、3 张 data:URI 内嵌图、CSP meta 在位、无 unsafe-inline |
| 9 | doctor 真实冒烟 | exit 0；许可记录含 G3/G4/G8 缺口+cleared_for=internal-eval；CC-BY 双引用署名（Jiang et al. 2023 / Kroner et al. 2020）端到端输出 |
| 10 | ruff 全项目 | All checks passed（line-length=120 口径） |
| 11 | 运行实测（C2 交付，L2 核验） | 冷启动 277-309ms；热推理均值 252.6-268.0ms@1920×1080（CPU）；进程峰值内存 ≈392MB；ONNX 元数据运行时实读 |
| 12 | S0 评估管线自检门 | 合成数据通过（uniform NSS=0/AUC=0.5/IG_U=0、CB IG_CB=0）；破坏容差→EvalGateError 失败退出有测试 |

## 失败案例记录

- 无假成功路径发现：无效 AOI 不留部分产物；输出目录冲突不覆盖历史；模型缺失/哈希不符结构化报错（C2 单测 7 类拒绝）。
- 过程事件（已闭环，不影响产物）：C1 ruff --fix 一次越界波及 C3 测试文件（约 5 处语义中性自动修复），C3 逐行复核确认无损；规则已固化（lint 自动修复必须显式限定独占文件清单）。

## 未验证项

- 游戏 UI 预测有效性：无眼动数据，未验证（validation-plan §4.2）。
- 公开数据基准评估（G2）：UEyes 12.9GB 下载待一级总控批准；评估脚本仅合成数据自检通过，表 B/C/D 未经真实数据消费。
- HTML 圈选交互：静态断言+客户端校验镜像测试通过，未做真实浏览器 E2E。
- 1s/7s 窗口敏感性权重：未下载（中体量，L2 将在评估开工前批准）。
- 跨设备一致性：仅本记录环境实测。

## 结论

**G1 工程闭环：通过。** CLI 成功/错误分支、概率与坐标正确性、截图到报告实际跑通均以真实执行证据闭合；产物明确标注模型预测语义与许可门禁。G2（比较可信）待 UEyes 批准后由评估脚本在公开数据上运行；G3 剩余项为 Skill 封装（即将派发）与 A/B 色阶的浏览器端人工抽查。
