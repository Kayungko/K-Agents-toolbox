# 来源与选型记录

文档日期：2026-09-11。下列事实来自方案讨论期间读取的官方仓库、厂商文档和论文页面；本项目未进行模型安装或推理测试。链接可能随上游更新，正式选型时固定 commit 和资产哈希。

## 1. 公开来源

| 来源 | 已读取信息 | 对本方案的影响 |
| --- | --- | --- |
| [DeepGaze 官方仓库](https://github.com/matthias-k/DeepGaze) | IIE 空间 log density、显式中心偏置；MSDB 要求 pixels per degree；III 使用注视历史 | 首版考虑静态密度，不从空间图推断时间；先验和观看配置需记录 |
| [DeepGaze setup.py](https://github.com/matthias-k/DeepGaze/blob/main/setup.py) | torch/torchvision 等依赖；MIT 分类与 license 字段被注释 | 不能据此宣称项目已按 MIT 授权 |
| [DeepGaze IIE 实现](https://github.com/matthias-k/DeepGaze/blob/main/deepgaze_pytorch/deepgaze2e.py) | 多骨干组合、预训练权重加载 | 需要完整依赖与权重清单；不得将其视为无需测量的轻量模型 |
| [DeepGaze 许可 Issue #15](https://github.com/matthias-k/DeepGaze/issues/15) | 商业使用许可问题仍为打开状态 | 正式集成与分发前明确许可；不将“公开”写成“可自由商用” |
| [Attention Insight 功能](https://attentioninsight.com/features/) | 热图、AOI 占比、焦点图与清晰度功能 | 可作为产品能力参考；不直接复制其专有分数语义 |
| [Attention Insight API](https://attentioninsight.com/api/) | 可将设计评估接入产品 | 外部后端候选 |
| [Attention Insight MCP](https://app.attentioninsight.com/mcp/instruction) | 官方远程 MCP、账户认证和分析额度 | Agent 集成候选；不是已接入的能力 |
| [Neurons 热图说明](https://knowledge.neuronsinc.com/neurons-ai-heatmaps-and-interpretation) | 多种预测图与 AOI 含义 | 不混淆注意力、复杂度和其他厂商指标 |
| [Neurons 插件说明](https://knowledge.neuronsinc.com/neurons-ai-plugins) | Figma 和浏览器集成 | 后续交互流程参考 |
| [UEyes：CHI 2023](https://doi.org/10.1145/3544548.3581096) | 不同 UI 类型上的视觉显著性研究与模型比较 | 通用图片模型的表现不能直接推广到游戏 UI |

厂商的准确率宣传未作为本项目指标或承诺。外部链接引用用于说明来源，不代表已获得其实现、数据或模型的分发许可。

## 2. 拟定决策

| ID | 决策 | 理由 |
| --- | --- | --- |
| D001 | 独立项目，首版 CLI + Skill | 可以先验证计算与评审，再按需要接宿主 |
| D002 | 静态完整画面为首版范围 | 减少时序和任务条件的不确定性 |
| D003 | 模型计算与语义评审分离 | 数值可追溯；目标不会人为篡改热图 |
| D004 | DeepGaze 仅为候选 | 许可与实际运行、游戏效果尚未闭合 |
| D005 | 输出预测概率占比、面积与相对密度 | 避免大面积区域占比高被直接判为问题 |
| D006 | 不使用统一 UI 质量分数 | 缺少跨场景校准依据 |
| D007 | 原始浮点概率图为统计来源 | 不从伪彩图反推数值 |
| D008 | A/B 固定配置及共用色阶 | 避免模型或展示配置变化造成假差异 |
| D009 | 后端失败明确返回错误 | 不静默降级、替换模型或上传截图 |
| D010 | 接入设计工具放到后续 | 首先完成最小分析闭环 |

## 3. 尚待确定

- 正式后端及其代码、权重、骨干与依赖许可。
- 仓库原创内容的许可证；本次不代替维护者选择分发条款。
- 可用硬件、锁定依赖、可接受输入尺寸与运行成本。
- 初始中心偏置配置及观看条件假设。
- 有使用许可的游戏 UI 试用样本与标注。
- 是否有独立眼动数据，及目标评估场景。

## 4. 更新原则

模型被替换、配置改变或获得新实测证据时，更新此记录和项目状态，并使旧报告仍能追溯其原始版本。运行成功、人工认可和真实眼动验证分别记录。
