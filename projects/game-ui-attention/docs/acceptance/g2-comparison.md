# G2 比较可信验收记录

验收人：二级总控（L2）。验收日期：2026-09-12。模板依据：[validation-plan.md](../validation-plan.md) §6；评估协议：[research/benchmark-protocol.md](../research/benchmark-protocol.md)（r2-v0.1）。

## 版本与范围

- 评估代码：提交 `07b643a`（驱动+批准口径）→ `9e79221`（NSS/CC 常数图防护修复）→ `439b9b6`（指纹字节稳定修复）；第 3 轮终跑工作树 = 439b9b6 内容。
- 数据：UEyes（Zenodo 8010312，CC-BY-4.0），zip md5 `c2d53e6af0a47e1f459416d6839ec2c1`（每轮实跑前真实重验，L2 独立复算一致）。
- 后端：foveacast-onnx-{1s,3s,7s}-v1（权重 sha256 与 release digest 逐字一致）+ deepgaze-iie-v1（仅内部评估对照，gaps=[G1,G2,G5]）+ uniform/center_bias 基线。
- 范围：公开数据（通用 UI）上的模型比较可信度。**不含游戏 UI 有效性**（无眼动数据，validation-plan §4.2）。

## 最终冻结指纹（第 3 轮终跑，权威值）

| 项 | 值 |
| --- | --- |
| 官方划分 splits.v1.json | sha256 `a5aa15a2521be1d6d51d6da10b4a832b073b2decfb1c5f7acfbe617eeefc3576`（Train 1872/Test 108，来源 image_types.csv） |
| 备选划分 splits.v2-fallback.json | sha256 `a02693b8702bcac38500f21cf175e78dea0b1345e80f9e42eecb7a5ba412859f`（Train 1335/Val 337/Test 308，dHash 簇约束整簇移动 109 图，cross_split=0） |
| CB 基线 v1（1s/3s/7s） | `8dd0d732…`/`ab93bee9…`/`2c75b857…`（train-only，64×64，σ_bin=1，计数加权） |
| 评估配置 config_hash | `99fb7a6051ae6479…`（三轮一致；S1-S3 启用、S2 硬门槛、L1 批准 2026-09-11 注记入哈希） |
| 三轮哈希链说明 | 第 1 轮 splits=`bb9fac09…`、第 2 轮=`01249333…`、第 3 轮=`a5aa15a2…`——前两轮文件内嵌生成时间戳致同内容哈希漂移（划分条目逐字一致），第 3 轮起指纹字节稳定（3 项字节级确定性回归测试守护）；本表为唯一权威引用值 |

## 实跑链（第 3 轮终跑，exit 0，全部真实执行）

zip md5 重验 → 划分冻结 → dHash64 近重复聚簇+泄漏审计五查 → 554 有效日志×3 窗口真值重建（BPOGV=1、TIME 前缀截断、1920×1200 letterbox 逆变换、越界丢弃计数）→ CB train-only×3 → S0 门 → 108 test 图×4 profile 推理（skipped=[]）→ 逐图 CSV（4536 行×3 窗口）→ bootstrap（B=2000、种子 20260911、类别分层）→ 配对差值 CI+胜率+Holm（54 行）→ 表 A-E → S0-S4 判定；备选划分 308 图同链独立重跑 → 官方 vs 备选对照报告（严格分表）。

## 判定结果（批准口径原文=S2 每窗口硬门槛；数字为第 3 轮终跑，与前两轮判定一致）

| 判据 | 结果 | 关键数字 |
| --- | --- | --- |
| S0 管线自检 | **pass** | uniform NSS/IG_U=0.0000 [0,0] 精确、sAUC/AUC-Judd=0.5000 [0.5,0.5] 精确、CB IG_CB=0；uniform IG_CB 与 CB IG_U 镜像 ∓0.8542@1s |
| S1 基线有效性 | **fail（边际）** | 6 子项过 5；唯 3s/sAUC CB−uniform ci_low≈-0.0005 跨零。诊断（L2 核准）：CB 的 IG_U/NSS 三窗口全强、1s/7s sAUC 过——3s 窗口 shuffled-fixation 判别增益真实微小，非构造/真值管线缺陷 |
| S2 候选及格线（硬门槛） | foveacast-1s **pass**（6/6）；foveacast-3s **fail**；foveacast-7s **fail**；deepgaze-iie **fail** | 3s 唯一未过项=1s/IG_CB ci_low≈-0.050（胜率 53.8%，边际）；其窗口匹配行 3s×3s：IG_CB +0.552 [0.474,0.630]、NSS +0.630、胜率 92.6%；7s 模型 7s×7s：IG_CB +0.540、胜率 96.3%；DeepGaze IG_CB 三窗口全负（-0.96/-0.51/-0.22） |
| S3 跨域一致性 | not_applicable | FiWI 不纳入（许可未核实，L1 批准口径） |
| S4 相对水平 | disabled | UMSI++ 不跑；作者报告值仅文献参考，未混表 |

**描述性观察（不作选型结论）**：窗口匹配单元格全过且幅度大；交叉窗口（→1s）一致退化，与"模型按窗口微调、真值前缀嵌套"构造一致；**UI 微调增量价值量化成立**：3s/IG_CB 上 foveacast-3s（+0.552）vs DeepGaze IIE（-0.512），差约 1.06 bits——通用域模型在 UI 数据上劣于数据驱动 CB，UI 微调显著优于通用域。

**S2 口径裁决提请（L1，选型层面）**：批准原文=每窗口硬门槛→主候选 3s 判 fail（边际交叉窗口项）；窗口匹配口径下 3s 强过。L2 技术建议修订为窗口匹配口径（依据如上构造性解释+备选划分下 1s/IG_CB 转正）。裁决前一切记录按原文口径，选型结论不提前生效。

## 泄漏审计与对照结论

- 官方划分：41 个近重复簇中 **14 簇跨 train/test**（含 39/44 图大簇；Hamming≤8 冻结阈值）——不满足协议 §3.4.1，如实记录未改划分，明细在 audit.json/表 D。
- 备选划分（簇约束，cross_split=0）对照：**官方 test 分数未因泄漏虚高**——候选模型 IG_CB/NSS 差值 diff(O−F) 多为负（备选下更高，如 foveacast-3s 3s/IG_CB 0.552→0.678；官方边际未过的 1s/IG_CB 0.049→0.152 转正）。机制：官方 CB 的 train 含 test 近重复→抬高 CB 自身→压低候选相对 IG_CB。结论：官方划分结果保守方向可信；边际 fail 项系测试集构成噪声而非模型缺陷（描述性，组间非配对）。
- **备选划分 S2 描述性判定**（非正式口径，仅供 L1 裁决参考）：foveacast-3s **pass 6/6**（1s/IG_CB +0.152 [0.101,0.201]）、foveacast-1s pass、foveacast-7s fail（1s 窗 -0.071）、deepgaze-iie fail——官方划分下 3s 主候选的唯一边际 fail 在无泄漏划分中翻转为 pass，且完全由 1s/IG_CB 单元格驱动；与"窗口匹配行全过、交叉窗口退化"模式相互印证。

## 排除与数据事实

- 1s 窗口 test 侧 17/108 low_fixation 排除（指标基于 91 图）；train 侧 75 low_fixation+4 low_viewers；3s/7s 零排除。两侧分列入表 D。
- 图像模式：RGB 1664/RGBA 304（真透明 6 张 ID 在案）/P 10/L 1/CMYK 1；RGBA 按官方 cv2 口径丢 alpha（不合成不填黑白）；分析 CLI 路径保持更严格透明拒绝口径，两路径不混用。
- 勘误：划分权威文件为 image_types.csv（协议所写 info.csv 在 zip 内不存在；分隔符 ';'，列 Image Name/Category/Block/Train/Test）。

## 缺陷披露与修复记录（validation-plan §6 失败案例栏目）

1. **NSS/CC 常数图防护缺陷**（C1 自查发现，主动披露）：`std==0` 判定在大数组（667×1110）上因 ulp 级舍入失效，uniform 基线逐图 NSS 出现 ±1.0 噪声（46/108 图@官方 1s）。影响界定：仅 uniform NSS 行与 S1 CI 宽度；S1 fail 由 sAUC 驱动（秩统计不受影响）、S2 完全不受影响、其余指标不受影响、两轮判定均不翻转。修复：逐元素 max==min 严格判定+S0 大网格防回归门+2 回归测试（`9e79221`）。修复后 uniform 全窗口精确 0.0000 [0,0]。
2. **指纹时间戳漂移**（L2 取证与 C1 自查独立同发现）：splits 内嵌 generated_at_utc+np.savez 写 zip 当前时间→同内容跨重跑哈希漂移，损害审计可核性。修复：移除时间戳字段（生成时刻移记 summary）+CB 手工 zip 固定条目时间戳+3 项字节级确定性回归测试（`439b9b6`）。第 3 轮终跑固化稳定指纹。
3. 处置模式：两次均为"自查→主动披露→精确影响界定→先修复后重跑→回归测试固化"，全套件每轮复绿（527→533→535 passed，L2 每轮独立复跑一致）。

## L2 独立复核清单（全部真实执行）

- 三轮套件复跑：527/533/535 passed 与各轮自报一致；
- 权重/数据哈希独立复算：foveacast×3、deepgaze2e.pth、UEyes zip md5 全部逐字匹配；
- splits/CB 哈希复算与时间戳根因取证（读文件内嵌元数据定位）；
- 表 A×summary 交叉抽验逐字一致；uniform 行精确值核验；
- 判定不翻转核验（三轮 S0/S1/S2 一致）；重跑作业全程进程级健康监控。

## 未验证项

- 游戏 UI 预测有效性（无眼动数据，维持未验证；本记录全部结论限公开通用 UI 数据）。
- S2 口径修订（窗口匹配）待 L1 裁决；裁决不改变本记录数字，仅改变选型结论表述。
- 时长加权敏感性行未跑（批准口径默认计数加权）。
- 参与者级不确定性未建模（协议 §7.2 最小方案，限制声明入产物）。
- FiWI 跨域对照未执行（许可未核实）。

## 结论

**G2 比较可信：通过。** 固定测试划分（官方冻结+备选对照双轨）、同窗口同预处理（config_hash 三轮一致）、中心偏置基线（train-only+版本化哈希）、逐图指标与分组误差（bootstrap CI+配对差值+Holm）全部落地；泄漏发现如实审计并量化（方向：官方分数保守）；文献值未混表；两次缺陷披露-修复-重跑链完整可追溯。模型选型表述以批准口径原文为准（S2：仅 1s 模型全窗口通过；3s 主候选窗口匹配强过、口径修订待 L1 裁决）。产物：local-data/eval-output/（官方）与 fallback/（备选+对照），均忽略目录；本记录引用第 3 轮冻结指纹。
