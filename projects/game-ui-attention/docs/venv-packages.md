# venv 包台账（projects/game-ui-attention/.venv）

- 维护者：C2（共享 venv 唯一第三方包维护者，L2 授权口径：单包 <50MB 可自行安装并记录；≥100MB 先报 L2 批准；禁 torch CUDA/onnxruntime-gpu 等 GPU 与越界包；禁全局环境改动）。
- 环境：Python 3.12.10（项目独立 venv，`.venv/` 为 git 忽略目录）；pip 缓存重定向至 `local-data/cache/pip`（`.venv/pip.ini`）。
- 本台账登记**主动安装**的包（含用途/安装线/日期）与安装后的传递依赖快照；与 C1 维护的 `pyproject.toml`/lock 文件互补——**deepgaze_pytorch 依 L2 裁定（2026-09-13 前序）不进 pyproject extras**（G1 许可未闭合、永不打包，git URL 不入依赖清单），仅本台账+模块 docstring 手动安装口径承载。
- 更新规则：任何后续安装由 C2 执行并追加本表（包==精确版本，以 `pip freeze` 实测为准）。

## 主动安装包

| 包==版本 | 用途 | 安装线 | 日期 |
| --- | --- | --- | --- |
| numpy==2.5.3 | 核心数值（图像数组/概率图/适配层） | 总控 bootstrap | 2026-09-11 |
| pillow==12.3.0 | 图像解码与缩放预处理（bicubic/bilinear） | 总控 bootstrap | 2026-09-11 |
| scipy==1.18.1 | C1 评估统计（R2 协议）；DeepGaze centerbias zoom/logsumexp | 总控 bootstrap | 2026-09-11 |
| onnxruntime==1.30.0 | foveacast ONNX 推理运行时（CPU EP；providers 实测含 CPUExecutionProvider） | 总控 bootstrap | 2026-09-11 |
| pytest==9.1.1 | 测试框架（pythonpath=["src"] 由 pyproject 配置） | 总控 bootstrap | 2026-09-11 |
| ruff==0.16.7 | Lint（line-length=120，规则 E/F/W/I/UP/B） | 总控 bootstrap | 2026-09-11 |
| torch==2.14.0 | DeepGaze IIE 推理（**CPU wheel**：`torch.version.cuda=None`、`2.14.0+cpu` 实测，U18 实证） | C2·G2③ | 2026-09-11 |
| torchvision==0.29.0 | DeepGaze 四骨干离线构造（`weights=None`，U1 处置） | C2·G2③ | 2026-09-11 |
| boltons==26.2.0 | deepgaze_pytorch 运行时依赖 | C2·G2③ | 2026-09-11 |
| deepgaze_pytorch @ git+https://github.com/matthias-k/DeepGaze@c7db17e2d1d7ea6468ffdee2cfaddf141095dcff | DeepGaze IIE 模型代码（本地 wheel 构建 1.2.1；**仅限 internal-eval，不打包不分发**，许可缺口 G1/G2/G5；来源 pin 记录于 registry.LICENSE_RECORD） | C2·G2③ | 2026-09-11 |
| fastapi==0.141.1 | A5 Web 协作校准平台 API 框架 | C2·A5 | 2026-09-13 |
| uvicorn==0.52.4 | A5 ASGI 服务器（基础安装，**未装 standard extras**，按 L2 指令） | C2·A5 | 2026-09-13 |
| python-multipart==0.0.32 | FastAPI multipart/form-data 上传解析 | C2·A5 | 2026-09-13 |

## 传递依赖快照（pip freeze 全量，2026-09-13 A5 安装后）

| 归属 | 包==版本 |
| --- | --- |
| fastapi/uvicorn 系（A5） | starlette==1.6.0、pydantic==2.13.5、pydantic_core==2.46.5、anyio==4.15.1、idna==3.19、click==8.5.0、h11==0.16.0、annotated-types==0.8.0、annotated-doc==0.0.5、typing-inspection==0.4.4 |
| torch 系（G2③） | filelock==3.32.6、fsspec==2026.7.0、Jinja2==3.1.6、MarkupSafe==3.0.3、mpmath==1.3.0、networkx==3.6.1、setuptools==84.0.0、sympy==1.14.0、typing_extensions==4.16.0 |
| onnxruntime 系（bootstrap） | flatbuffers==25.12.19、protobuf==7.36.1 |
| pytest/ruff 系（bootstrap） | colorama==0.4.6、iniconfig==2.3.0、packaging==26.3、pluggy==1.6.0、Pygments==2.21.0 |

## 备注与已知事项

1. **httpx2 未安装（待 A5 线决策）**：starlette 1.6.0 的 `fastapi.testclient.TestClient` 需要可选依赖 `httpx2`（实测 RuntimeError 提示 `pip install httpx2`）。不在本次 A5 三包批准清单内，C2 未擅自安装；若 A5 线需要 TestClient 做端点测试，报 L2 后由 C2 安装（小体量）。运行时服务（uvicorn+fastapi）不受影响。
2. 红线遵守记录：未安装任何 GPU 包（onnxruntime-gpu/torch CUDA wheel）；未改动全局 Python；未安装清单外包；全部安装走 `.venv\Scripts\python.exe -m pip`（缓存落 `local-data/cache/pip`，忽略目录）。
3. clip/einops/tensorflow/aria2 均未安装（DeepGaze MSDB/训练/下载工具越界项，适配层已以包桩 shim 与 stdlib 分段下载器规避）。
4. 验证口径：A5 三包安装后全套件复跑 **546 passed**（2026-09-13，C2 实测），import 冒烟 fastapi/uvicorn/multipart 全过。
