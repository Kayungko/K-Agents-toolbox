# 运行环境可行性研究（R3）

- 核查日期：2026-09-11（环境实测快照约为当日 17:05 本地时间；网络来源均于当日核查）。
- 任务性质：只读实测 + 设计建议。本次未安装任何依赖、未下载任何权重/数据集、未创建 venv 或目录实体、未修改除本文档外的任何文件。
- 证据分级标注：`已验证` = 本次有直接命令输出或官方页面证据；`推断` = 由证据合理推导但未实测；`未验证` = 缺少证据，注明原因。
- 脱敏说明：本文可公开。已去除用户名、机器名、个人绝对路径与本机运行应用清单；保留盘符、OS 版本与硬件型号。原始命令输出摘录见附录 A（已脱敏）。

## 0. 结论摘要

1. **本机完全具备运行全部候选路线的硬件条件**：RTX 4070（12 GB 显存）+ i7-14700KF（20 核 28 线程）+ 64 GB RAM + D 盘 1 TB 可用空间（均已验证）。
2. **最低摩擦起步路线是 foveacast ONNX**：onnxruntime CPU（14.7 MB）+ 单个 UI 微调 MSI-Net ONNX 权重（FP16 53.9 MB，release 页自带 sha256），合计 <100 MB，全部为小体量档，第二波无需任何下载审批即可开工（已验证）。
3. **DeepGaze IIE 路线成本更高**：权重 400 MB（中体量）+ torch CPU 124.1 MB（中体量）+ 4 个骨干经 `torch.hub` 自动下载的 ImageNet 权重（估算合计 300–450 MB，未验证）；且源码使用 `pytorch/vision:v0.6.0` 旧 hub tag 与 `pretrained=` 旧 API，与当前 torch 2.14/torchvision 0.29 的兼容性**未验证**，可能需要适配补丁；代码许可未闭合（Issue #15 打开，见来源记录）→ 存在许可门禁。
4. **大体量项（需一级总控批准）**：UEyes 数据集 12.9 GB（Zenodo，CC-BY-4.0，已验证）；torch Windows CUDA wheel（估算 2–3.5 GB，精确值未验证）。
5. **Python 建议**：venv 基于 **3.12**（本机已有 Store 版；foveacast-training 官方 pin 3.12）；本机默认 3.14.2 亦可行（torch/onnxruntime 均有 cp314 wheel，已验证），但老代码库（DeepGaze）配新 Python 未知数更多。
6. **磁盘风险**：C 盘仅剩 60 GB（已验证），而 pip/torch/HF 默认缓存均在 C 盘用户目录 → 第二波必须按 §2.2 将缓存重定向到 D 盘项目内忽略目录。
7. **无系统级 CUDA Toolkit/cuDNN**（nvcc 不在 PATH、CUDA_PATH 未设置，已验证）→ 走 pip wheel 自带运行时路线；onnxruntime-gpu 首版不选（Windows 需要系统级 CUDA/cuDNN DLL，安装即改全局环境，越界）。
8. **.gitignore 现状足够**：`.venv/`、`**/model-cache/`、`**/runs/`、`**/local-data/` 均实测命中（git check-ignore 证据见 §2.3）；建议追加 4 条扩展名兜底规则（原文见 §2.3，供总控决定，本任务未改动）。

## 1. 本机环境实测（2026-09-11）

全部为只读查询命令（PowerShell/CIM、nvidia-smi、py、where.exe、git），完整命令与脱敏输出见附录 A。

| 项目 | 实测结果 | 状态 |
| --- | --- | --- |
| OS | Microsoft Windows 10 IoT 企业版 LTSC，Version 10.0.19044（Build 19044），64 位（注：不是 Windows 11） | 已验证 |
| CPU | Intel Core i7-14700KF，20 核 / 28 逻辑处理器（混合架构），WMI 报告 MaxClockSpeed 3400 MHz（基准值，睿频不在该字段体现） | 已验证 |
| RAM | 总物理内存 68,482,342,912 B ≈ 64 GB；可见 66,877,288 KB ≈ 63.8 GiB；快照时可用 18,908,704 KB ≈ 18.0 GiB | 已验证 |
| GPU | NVIDIA GeForce RTX 4070（WDDM 模式），显存 12,282 MiB ≈ 12 GB；快照时已占用 4,787 MiB（桌面常驻应用较多），空闲 38 °C、P8、13 W / 200 W；驱动版本 596.36；nvidia-smi 报告 CUDA Version 13.2（= 驱动支持上限，非已装 Toolkit） | 已验证 |
| GPU（WMI 口径） | Win32_VideoController.AdapterRAM 显示 ≈ 4 GB，为 32 位字段截断，以 nvidia-smi 的 12,282 MiB 为准；另存在一个虚拟显示适配器（非 NVIDIA，不参与 CUDA 计算） | 已验证 + 推断（截断原因） |
| Python | py 启动器列出：3.14*（默认，用户级安装于 `C:\Users\<user>\AppData\Local\Programs\Python\Python314`）与 3.12（Microsoft Store 版，WindowsApps）；`python --version` = Python 3.14.2 | 已验证 |
| git | git version 2.52.0.windows.1 | 已验证 |
| 磁盘 | C: 总 778.3 GB / **可用 60.0 GB**；D: 总 1863 GB / **可用 1024 GB**（仓库所在盘）；E: 可用 324.3 GB；G: 可用 877.6 GB | 已验证 |
| CUDA Toolkit / cuDNN | `nvcc` 不在 PATH；`CUDA_PATH` 环境变量未设置；`C:\Program Files\NVIDIA*` 下无 CUDA Toolkit 目录 → **系统级 Toolkit/cuDNN 未安装**。PyTorch 的 pip CUDA wheel 自带 CUDA 运行时与 cuDNN DLL，仅需驱动即可运行（此句为通用机制，推断，第二波以 `torch.cuda.is_available()` 实测为准） | 已验证（未安装事实）|
| 仓库状态 | main 分支，工作树干净（`git status --porcelain` 空输出），HEAD = b3ddb20（wave-1 研究文档提交） | 已验证 |

**对方案的关键含义**：

- 显存 12 GB 对所有候选模型都应充裕（各模型峰值显存**未验证**，见 §5-U4）；但快照显示桌面应用常驻占用 ≈4.8 GB，推理时可用显存 ≈7.4 GB，`doctor` 命令应实测空闲显存再决定 device，`GPU_OOM` 错误路径（data-contract §7 退出码 4）必须实现。
- CPU 20 核 + 18 GB 可用内存对 CPU 推理足够（延迟**未验证**，见 §5-U5）。
- C 盘 60 GB 是主要磁盘风险点：torch CUDA wheel 下载 + pip 缓存 + 解压安装可能一次吃掉 5–8 GB 且默认都落 C 盘 → §2.2 的缓存重定向为必做项。

## 2. 隔离环境设计（只设计，未创建）

### 2.1 项目专用 venv

- **位置**：`projects/game-ui-attention/.venv/`。
- **忽略核实**：仓库 `.gitignore` 第 9 行 `.venv/` 为无前导斜杠模式，匹配任意层级同名目录；`git check-ignore -v` 实测 `projects/game-ui-attention/.venv/pyvenv.cfg` 命中该规则（附录 A.6）。**无需新增规则**。
- **创建命令（第二波执行，本任务未执行）**：

```powershell
# 在仓库根目录
py -3.12 -m venv projects\game-ui-attention\.venv
# 使用（免激活、绝对路径调用，避免污染全局 shell）：
projects\game-ui-attention\.venv\Scripts\python.exe -m pip --version
```

- **Python 版本取舍**：

| 选项 | 依据 | 结论 |
| --- | --- | --- |
| 3.12（Store 版，已装） | foveacast-training 官方 pin 3.12（README 已验证）；torch 2.14 / torchvision 0.29 / onnxruntime 1.30 全部支持（PyPI 页面已验证）；生态最稳 | **推荐** |
| 3.14.2（本机默认，已装） | torch 2.14 有 cp314/cp314t win_amd64 wheel（已验证）；onnxruntime 1.30 支持 3.11–3.14（已验证）；但 numpy/scipy/pillow 的 cp314 wheel 未逐一核查，DeepGaze 老代码 × 新 Python 未知数更多 | 备选 |
| 新装其他版本 | 违反"禁改全局环境"边界 | 不采用；若 Store 版 3.12 建 venv 有缺陷（未验证，§5-U15），报总控裁决 |

- 注：torchvision 0.29.0 声明 `Requires Python !=3.14.1, >=3.10`（PyPI 已验证）——3.14.1 被上游排除，本机 3.14.2 不受影响，但说明 3.14 线仍有上游修补期，进一步支持选 3.12。

### 2.2 模型权重与缓存目录（全部位于忽略目录内，不入公开仓库）

与 technical-design §10"运行目录、截图与权重默认不纳入 Git"及 orchestration-plan §7"缓存与生成物使用忽略目录"一致。以下路径均经 `git check-ignore` 实测被忽略（附录 A.6）：

```text
projects/game-ui-attention/
├── model-cache/            # 权重与框架缓存（.gitignore:25 **/model-cache/）
│   ├── foveacast/          #   foveacast-v3-*.onnx + parity/quality json
│   ├── deepgaze/           #   deepgaze2e.pth、centerbias_mit1003.npy
│   ├── torch-hub/          #   TORCH_HOME（torch.hub 克隆与 checkpoints）
│   └── hf/                 #   HF_HOME（若走 MSI-Net HF 快照路线）
├── local-data/             # 数据与下载缓存（.gitignore:24 **/local-data/）
│   ├── ueyes/              #   UEyes 解压数据（若获批）
│   ├── screenshots/        #   分析输入截图（含内部截图，绝不入 Git）
│   └── cache/pip/、cache/uv/  # 包管理器下载缓存
└── runs/                   # 运行目录 <run-directory>（.gitignore:26 **/runs/）
```

- **环境变量重定向（不改全局，只在项目 venv 激活脚本内设置）**——设计示例原文，第二波写入 `.venv\Scripts\Activate.ps1` / `activate.bat` 末尾（venv 本身被忽略，不进 Git；同时建议在 CLI 代码内做同样默认值兜底并写进文档，保证免激活调用也生效）：

```powershell
# 设计示例（未创建）：.venv\Scripts\Activate.ps1 追加段
$proj = Split-Path (Split-Path $PSScriptRoot)          # -> projects/game-ui-attention
$env:PIP_CACHE_DIR = "$proj\local-data\cache\pip"
$env:UV_CACHE_DIR  = "$proj\local-data\cache\uv"
$env:TORCH_HOME    = "$proj\model-cache\torch-hub"
$env:HF_HOME       = "$proj\model-cache\hf"
```

- **理由**：① C 盘仅剩 60 GB（已验证），而 pip/uv/torch/HF 默认缓存全在 C 盘用户目录；② D 盘剩 1 TB 且仓库在 D 盘；③ 全部落在已忽略目录内，公开仓库零泄漏风险；④ `TORCH_HOME` 重定向同时解决 DeepGaze IIE 的 torch.hub 骨干权重落点（§5-U10）。
- **注意**：实测发现 `projects/game-ui-attention/cache/`（裸 cache 目录）**不匹配任何现有规则**（附录 A.6 第 8 行）→ 缓存必须放在 `local-data/cache/` 或 `model-cache/` 之下，不要用裸 `cache/` 目录；或由总控采纳 §2.3 的 `*.whl` 兜底规则。

### 2.3 .gitignore 核实结果与建议追加规则

现状核实（`git check-ignore -v` 实测，命令与输出见附录 A.6）：

| 建议路径示例 | 是否忽略 | 命中规则（.gitignore 行号） |
| --- | --- | --- |
| `.venv/pyvenv.cfg` | 是 | `:9 .venv/` |
| `model-cache/deepgaze2e.pth` | 是 | `:25 **/model-cache/` |
| `model-cache/foveacast-v3-3s-fp16.onnx` | 是 | `:25 **/model-cache/` |
| `model-cache/umsi++.hdf5` | 是 | `:25 **/model-cache/` |
| `runs/run1/manifest.json` | 是 | `:26 **/runs/` |
| `local-data/ueyes/sample.png` | 是 | `:24 **/local-data/` |
| `work/tmp.txt` | 是 | `:27 **/work/` |
| `cache/pip/x.whl` | **否（缺口）** | 无匹配规则 |

结论：推荐布局（§2.2）已被现有目录规则**完整覆盖**，无必须追加项。存在一个兜底缺口：若权重/离线安装包散落在忽略目录之外（如误放 `src/` 或裸 `cache/`），`*.onnx`、`*.hdf5`、`*.whl` 等格式不在现有扩展名规则（仅 `*.pth/*.pt/*.ckpt/*.safetensors`）内。**建议总控追加以下规则原文（R3 未改动 .gitignore）**：

```gitignore
# Model interchange & offline package formats outside ignored dirs (R3 建议)
*.onnx
*.hdf5
*.h5
*.whl
```

### 2.4 依赖策略

**路线取舍（CPU vs CUDA）**：

| 路线 | 依赖 | 体量 | 判断 |
| --- | --- | --- | --- |
| A. foveacast ONNX（推荐先行） | onnxruntime CPU 1.30.0 + numpy + pillow，**无需 torch** | ≈80–100 MB（小体量） | 输入 240×320，CPU EP 预期足够（延迟未验证 §5-U5）；release 自带 sha256 可直接进 doctor 校验 |
| B. DeepGaze IIE（CPU torch） | torch 2.14.0 CPU win wheel（124.1 MB，推断为 CPU 构建，§5-U18）+ torchvision 0.29.0（1.4 MB）+ scipy + 权重 400 MB + hub 骨干权重 ≈300–450 MB（估算） | ≈850 MB–1 GB（中体量，报总控） | 4 骨干 ×30 组件集成（源码已验证），CPU 延迟风险最高；许可门禁未闭合 |
| C. torch CUDA wheel | download.pytorch.org/whl/cu128 或 cu130 | 估算 2–3.5 GB（未验证 §5-U11） | **仅当** B 路线 CPU 延迟不可接受时升级；>1 GB 需一级总控批准。驱动 596.36 支持 CUDA ≤13.2（已验证）→ cu130 及以下可用、cu134 不可用（推断） |
| D. onnxruntime-gpu | 需要系统级 CUDA/cuDNN DLL | 未核查 | **首版不选**：本机无 Toolkit（已验证），安装即改全局环境，越界；A 路线 CPU EP 已够 |

**版本 pin 与 lock**：

- `pyproject.toml` 直接依赖用精确 pin（`==`），与 technical-design §3"在后端跑通后锁定实际兼容的 Python 与依赖版本，不用上游未设上限的依赖声明代替兼容性验证"一致。2026-09-11 核查到的当前稳定版基线：**torch 2.14.0**（2026-09-02 发布）、**torchvision 0.29.0**（同日发布；与 torch 2.14 的配对为推断，官方配对表最高列到 2.13↔0.28）、**onnxruntime 1.30.0**（2026-09-10 发布）。
- lock 方案：**推荐 uv**（`uv lock` 生成 `uv.lock` 提交 Git；解析快、单文件、foveacast-training 同生态已用 uv，README 已验证）；备选 pip-tools（`requirements.in` → `pip-compile --generate-hashes` → `requirements.txt` 提交）。两者本次均**只推荐不安装**。
- extras 分组建议（由 C1 写入 pyproject，内容 C2 提案）：核心 = numpy + pillow；`[onnx]` = onnxruntime；`[deepgaze]` = torch + torchvision + scipy；`[dev]` = pytest + ruff；`[eval]` = scipy + scikit-image（对齐 foveacast-training 的分组习惯，便于其评估脚本复用）。
- **权重哈希策略**（对齐 data-contract §4 model 字段"全部权重哈希"）：foveacast release 页自带 sha256（已验证）→ 作为期望值写进后端 registry 常量；deepgaze2e.pth 官方未发布哈希 → 第二波下载后本地计算 sha256 登记进 registry 与 manifest。doctor 校验"实际文件哈希 == 登记哈希"，不匹配退出码 3（`MODEL_NOT_READY`）。

## 3. 最小 CLI 工程布局建议（只写布局，未创建代码）

src 布局，包名 `ui_attention`，入口 `ui-attention`。目录树（标注默认所有权：C1=契约/AOI/统计、C2=后端环境/推理、C3=报告/A-B 色阶）：

```text
projects/game-ui-attention/
├── pyproject.toml                  # C1 唯一负责人（extras 内容由 C2/C3 提案，C1 合入）
├── uv.lock（或 requirements.txt）  # C1（lock 生成与更新）
├── README.md                       # 总控/一级总控口径，子任务不写
├── .venv/                          # 忽略（已核实）；第二波 C2 建
├── model-cache/                    # 忽略；运行期 C2 独占写入
├── local-data/                     # 忽略；数据集/截图/下载缓存
├── runs/                           # 忽略；<run-directory> 运行产物
├── docs/research/                  # R1/R2/R3 报告（本文所在）
├── src/
│   └── ui_attention/
│       ├── __init__.py             # C1
│       ├── cli.py                  # C1：四命令入口、参数解析、stdout JSON 信封、退出码分支
│       ├── errors.py               # C1：结构化错误目录（MODEL_NOT_READY/INVALID_AOI/GPU_OOM/INVALID_DENSITY/COMPARISON_INCOMPATIBLE…）
│       ├── contracts/              # C1：AnalyzeRequest v1 / analysis v1 / comparison v1 / BackendInfo / PredictionResult schema 与校验（含后端接口签名定义）
│       ├── imaging.py              # C1：图片校验、方向/透明度处理、缩放与坐标变换记录
│       ├── aoi/                    # C1：矩形/多边形校验、掩码、嵌套重叠去重
│       ├── metrics/                # C1：log_density→概率归一化、probability_mass、area_fraction、relative_density、delta_pp；eval/ 子模块放公开数据评估脚本（IG/NSS/CC，R2 协议）
│       ├── backends/               # C2：registry.py（profile 登记）、doctor.py（设备/依赖/权重哈希/许可记录探测）、weights.py（下载+校验+缓存）、onnx_foveacast.py、deepgaze_iie.py（接口实现 C1 在 contracts/ 中定义的 describe()/predict()）
│       └── report/                 # C3：overlay.py（热图叠加 PNG）、colorscale.py（A/B 共用色阶，参数写入 manifest）、html/（report.html 模板，自包含、转义、默认禁外链）、compare_view.py、export_regions.py（区域圈选导出 JSON）
└── tests/
    ├── fixtures/                   # C1 维护：合成小图与 AOI 夹具（不含真实游戏截图）
    ├── unit/
    │   ├── test_contracts_*.py     # C1
    │   ├── test_imaging_*.py       # C1
    │   ├── test_aoi_*.py           # C1
    │   ├── test_metrics_*.py       # C1
    │   ├── test_backends_*.py      # C2（可用 tiny 假 ONNX/跳过标记隔离真权重）
    │   └── test_report_*.py        # C3
    └── integration/                # 跨线整合：二级总控串行执行（orchestration-plan §5）
```

**四个命令 → 模块 → 产物映射**（对齐 data-contract §1/§5/§6/§7）：

| 命令 | C1 承担 | C2 承担 | C3 承担 | 落盘产物（写方） |
| --- | --- | --- | --- | --- |
| `doctor` | cli 壳、JSON 信封、退出码 3 语义 | backends/doctor.py：依赖版本、设备与空闲显存、权重存在性与 sha256、许可核查记录（不打印凭据；模型下载是独立显式步骤） | — | 无落盘；stdout JSON 信封 |
| `analyze` | contracts 校验（退出码 2）→ imaging 预处理 → metrics 归一化 + aoi 统计 → analysis.json/regions.json/density.npy 落盘 → manifest.json **最后写**（表示计算完成） | backends.predict()：推理 + runtime 字段实测（设备、精度、耗时、峰值内存/显存）；失败抛 GPU_OOM/MODEL_NOT_READY（退出码 4/3） | overlay.png（固定透明度+可记录色阶）、report.html（转义、自包含、无自动网络请求） | manifest.json(C1)、analysis.json(C1)、regions.json(C1)、density.npy(C1)、overlay.png(C3)、report.html(C3) |
| `summarize` | 读取既有 density.npy（校验图片 SHA-256 匹配），aoi/metrics 重算，更新 regions.json/analysis.json；**不重新推理**（不调用 C2） | — | report.html 区域展示刷新、标注编辑导出（export_regions） | regions.json(C1)、analysis.json(C1)、报告更新(C3) |
| `compare` | 兼容性校验（模型/权重/预处理/先验/画布尺寸，不兼容退出码 6 并保留原因）、AOI 按稳定 ID 配对、delta_pp、新增/移除区域单列 → comparison.json | — | A/B 页面共用色阶渲染，色阶参数写入 manifest | comparison.json(C1)、A/B 报告(C3) |
| review.json / review.md | schema 校验（finding 结构、evidence_refs） | — | 报告内展示 | 由 Agent 生成写入运行目录；CLI 只校验与呈现，不生成数值 |

**所有权不重叠规则**：

1. 三条线各自只写自己标注的目录/文件；跨线只经由 `contracts/` 中冻结的接口（`BackendInfo`、`PredictionResult`、`describe()/predict()`、report 渲染函数签名）协作，接口签名由二级总控在第二波开工前一次性冻结。
2. 共享文件 `pyproject.toml`、`cli.py`、`errors.py`、`tests/fixtures/` 由 **C1 唯一持有**；C2 的 extras 依赖项、C3 的模板资源依赖以"提案 → C1/总控合入"方式进入，避免并行写冲突。
3. `model-cache/`、`.venv/` 运行期由 C2 独占写入（均在忽略目录，无 Git 冲突面）；`runs/` 由 CLI 运行时写入，谁执行谁创建，不预建。
4. `tests/integration/` 与跨模块联调由二级总控串行处理，不分配给单线。
5. C1 的 metrics/eval（R2 评估协议实现）只读 `local-data/ueyes/`（若获批），不依赖 C2/C3 代码路径。

## 4. 下载体量与资源估计（核查日期 2026-09-11，均查官方页面，未下载）

分级：**小 <100 MB 可直接用 / 中 100 MB–1 GB 报二级总控 / 大 >1 GB 需一级总控批准**（orchestration-plan §7）。

| # | 资产 | 来源 URL | 体量 | 许可线索 | 用途 | 分级 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | onnxruntime 1.30.0（win_amd64，cp311–cp314） | https://pypi.org/project/onnxruntime/#files | 14.3–14.7 MB（已验证） | MIT（PyPI 页面） | foveacast ONNX 推理运行时 | 小 |
| 2 | foveacast v0.2.0 ONNX 权重（1s/3s/7s × FP16/INT8） | https://github.com/khawkins98/foveacast-training/releases/tag/v0.2.0 | FP16 53.9 MB/个、INT8 30.3 MB/个，release 页逐文件带 sha256（已验证；2026-04-18 上传）。例：3s FP16 sha256 `842a23f9…585ef76e`（全文见 release 页） | 代码 MIT；训练数据 UEyes CC-BY-4.0，需署名 Jiang et al. 2023（README 口径） | UI 微调 MSI-Net 权重，A 路线主力 | 小（单文件）。三窗口 FP16 全下 ≈161.7 MB → 中；**建议首下仅 3s 一个文件** |
| 3 | torch 2.14.0 CPU（win_amd64，PyPI 默认 wheel） | https://pypi.org/project/torch/#files | 124.1 MB（cp311–cp314，已验证） | BSD 系（PyPI SPDX：Apache-2.0 AND BSD-2/3-Clause AND MIT 等） | B 路线（DeepGaze）推理框架 | **中 → 报二级总控** |
| 4 | torchvision 0.29.0（win_amd64） | https://pypi.org/project/torchvision/#files | 1.4 MB（已验证） | BSD | DeepGaze 骨干构建依赖 | 小 |
| 5 | torch Windows CUDA wheel（cu126/cu128/cu130 变体） | https://download.pytorch.org/whl/ （根索引已验证变体目录存在） | **精确值未验证**（wheel 索引页过长无法完整抓取）；按历史 Windows CUDA 捆绑 wheel 数量级估算 2–3.5 GB（推断） | 同 #3 | GPU 加速，仅当 CPU 延迟不可接受 | **大 → 需一级总控批准** |
| 6 | DeepGaze IIE 权重 deepgaze2e.pth | https://github.com/matthias-k/DeepGaze/releases/tag/v1.0.0 | 400 MB（已验证，release 资产页） | 仓库 setup.py 的 license 字段被注释、商用许可 Issue #15 打开（见 sources-and-decisions.md） | 通用图片显著性基线 | **中 → 报二级总控**；许可闭合前不打包不分发（门禁） |
| 7 | centerbias_mit1003.npy | 同上 release | 8 MB（已验证） | 同 #6 | 中心偏置先验（MIT1003 口径，须按 technical-design §5 标注来源） | 小 |
| 8 | DeepGaze IIE 骨干 ImageNet 权重（torch.hub 首次构建自动下载 ×4 骨干） | `torch.hub.load('pytorch/vision:v0.6.0', …, pretrained=True)`（源码已验证）→ download.pytorch.org/models/*（具体 URL 未逐一核查） | 估算 80–120 MB/骨干、合计 ≈300–450 MB（推断，未验证：4 个骨干逐一 URL/大小未核查） | torchvision 预训练权重许可随上游 | IIE 推理前置（即使 deepgaze2e.pth 已含微调权重，hub 构建仍会触发下载） | 中（合计推断）→ 报二级总控 |
| 9 | DeepGaze 代码仓库 | https://github.com/matthias-k/DeepGaze | <10 MB（推断，纯代码仓） | 同 #6 | IIE/MSDB 实现 | 小 |
| 10 | MSI-Net 原始 TF SavedModel | https://huggingface.co/alexanderkroner/MSI-Net | ≈100 MB（HF 页面标注 repo 100 MB，已验证） | MIT（HF 页面） | 仅训练复现/TF→PyTorch 权重移植需要 | 中 → **暂缓**（A 路线推理不需要） |
| 11 | TensorFlow（一次性权重移植依赖） | https://pypi.org/project/tensorflow/ | ≈500 MB 磁盘（foveacast README 口径，未逐项验证） | Apache-2.0 | weights-import 一次性步骤，用后可卸载 | 中 → **暂缓** |
| 12 | UEyes 数据集 | https://zenodo.org/records/8010312 | **12.9 GB** 单 zip（已验证；md5 `c2d53e6af0a47e1f459416d6839ec2c1`，v1 发布于 2023-06-06） | CC-BY-4.0（Zenodo 页面） | 评估基准（R2 协议的公开数据划分）/ 微调数据 | **大 → 需一级总控批准**（注意：Zenodo 仅单 zip，即使只用 108 张测试图也需整包下载） |
| 13 | UMSI++ 代码 + 权重 umsi++.hdf5 | https://github.com/YueJiang-nj/UEyes-CHI2023 （saliency_models/UMSI++/） | 代码轻量（推断）；**权重体量与分发位置未验证**（README 只说"把 umsi++.hdf5 放进 weights 文件夹"，未给下载源） | 未核查（R1 职责） | UEyes 配套模型候选 | 未验证 |
| 14 | DeepGaze MSDB 骨干权重（CLIP + DINOv2） | README 称 `DeepGazeMSDB(pretrained=True)` 自动下载；URL 未核查 | 未验证（CLIP/DINOv2 类骨干通常合计数百 MB，推断） | 待查（OpenAI CLIP=MIT、DINOv2=Apache-2.0 需 R1 核实具体权重条款） | MSDB 后续候选（且需 pixel_per_dva 观看条件） | 未验证 |
| 15 | numpy / pillow / scipy | 各自 PyPI 项目页 | 每项数 MB–数十 MB 量级（推断，未逐项核查 wheel 大小） | BSD / MIT-CMU / BSD | 基础依赖 | 小 |

**路线预算汇总**：

| 路线 | 组成 | 合计体量 | 审批需求 |
| --- | --- | --- | --- |
| A 最小 ONNX 推理（推荐先行） | #1 + #2（仅 3s FP16 单文件）+ #15 | ≈80–100 MB | 全小体量，第二波可直接执行 |
| B DeepGaze CPU | A + #3 + #4 + #6 + #7 + #8 + #9 | 追加 ≈850 MB–1.0 GB | 中体量 → 报二级总控；且受许可门禁（#6）约束 |
| C 完整评估 | B + #12（可选 + #10/#11） | 追加 12.9 GB+ | **大 → 需一级总控批准** |
| D GPU 加速（可选） | #5 | 追加 2–3.5 GB（估算） | **大 → 需一级总控批准**；仅当 CPU 实测不达标 |

## 5. 兼容性未知项（逐条标注，除注明外均为"未验证"）

| 编号 | 未知项 | 现有证据与风险 | 状态 |
| --- | --- | --- | --- |
| U1 | DeepGaze IIE × torch 2.14 / torchvision 0.29 可运行性 | 已验证源码：`RGBResNext50` 用 `torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)`；`pretrained=` 旧 API 在新版 torchvision 已移除、且 EfficientNetB5 在 v0.6.0 中不存在（两点均为推断，未本机运行）→ IIE 在最新依赖下大概率需要适配补丁或旧版 torch 降级 | **未验证（高风险项）** |
| U2 | deepgaze_pytorch 对 torch 版本的要求 | setup.py 依赖未设上限（sources-and-decisions 已记录）；已读的 deepgaze2e.py 无版本守卫；实际可跑组合未知 | 未验证 |
| U3 | Python 3.14 长尾兼容 | torch 2.14 cp314 wheel、onnxruntime 1.30 cp311–cp314 均已验证存在；numpy/scipy/pillow 的 cp314 wheel 未逐一核查；torchvision 排除 3.14.1（已验证） | 部分验证 |
| U4 | 显存/内存峰值 | IIE 为 4 骨干并行集成（结构已验证），1024px 级输入推理显存未实测；foveacast 240×320 CPU 峰值内存未实测；GPU 快照已被桌面应用占用 ≈4.8 GB/12.3 GB（已验证）→ doctor 必须实测空闲显存 | 未验证 |
| U5 | 推理延迟 | IIE CPU 单图耗时未知（30 组件 ×4 骨干，预期偏慢，推断）；foveacast ONNX CPU 耗时未知（输入小，预期秒级内，推断） | 未验证 |
| U6 | onnxruntime CPU EP 对 opset 17 FP16/INT8 模型的支持与性能 | FP16 在 CPU 上可能回退 FP32 计算（通用行为，推断）；INT8 为静态量化（作者口径）；本机未加载过该模型 | 未验证 |
| U7 | foveacast 模型对游戏 UI 的适用性 | 固定 240×320 输入，作者自述小元素（细小文字/图标）低于该分辨率无法分辨；UEyes 训练分布为 2020–2022 西方语言桌面/移动 UI，作者自述可能不适应暗色模式与非拉丁文字 → 对高分辨率、中文、游戏风格 UI 的效果未知；作者公布的提升幅度（CC/NSS +25–64%）本项目未复现 | 未验证（作者口径已记录） |
| U8 | foveacast ONNX 输入/输出元数据 | README 示例用张量名 `"input"`/`"output"`、输入 (1,3,240,320) float32 [0,255]、输出 (1,1,240,320) [0,1]（页面口径已验证）；本机未加载过模型文件，实际 metadata 是否一致未验证 → 后端实现必须运行时读取 session 元数据，不硬编码张量名 | 未验证 |
| U9 | ONNX 重导出路线是否现成 | export_onnx.py 随仓库提供、含 parity 校验（README 口径已验证）；本机未运行；若未来用自有游戏 UI 数据微调再导出，依赖该仓库的 Python 3.12 + uv 工具链组合 | 未验证 |
| U10 | torch.hub 的网络与 git 依赖 | IIE 首次构建需 git（本机 2.52.0，已验证）+ 网络克隆 `pytorch/vision@v0.6.0`，并把骨干权重下载到 TORCH_HOME → 离线部署必须预置 `model-cache/torch-hub/`（§2.2 重定向已设计）；克隆与下载行为本机未运行 | 机制已验证（源码），行为未验证 |
| U11 | torch CUDA wheel 变体可用性与体量 | 驱动 596.36 → CUDA ≤13.2（nvidia-smi 已验证）；download.pytorch.org 存在 cu126/cu128/cu129/cu130/cu132/cu134 目录（根索引已验证）→ cu130/cu132 应可用、cu134 不可用（推断）；Windows CUDA wheel 精确体量与 sm_89（Ada）架构支持未验证（索引页过长无法完整抓取） | 未验证 |
| U12 | onnxruntime-gpu 的 Windows 前置条件 | 公开文档口径：Windows CUDA EP 需要系统级 CUDA/cuDNN DLL；本机无 Toolkit（nvcc/CUDA_PATH 缺失，已验证）→ 装 Toolkit 即改全局环境，越出 orchestration-plan §7 边界 → 首版不选该路线 | 未验证（决策依据为已验证的本机状态） |
| U13 | UMSI++ 可用性 | conda environment.yaml + TensorFlow + Jupyter 工作流（子目录 README 口径已验证）；`umsi++.hdf5` 权重的分发位置与大小未验证（README 未给下载源）；Windows/新 TF 版本兼容性未验证；environment.yaml 可能 pin 旧 TF/CUDA 组合（推断） | 未验证 |
| U14 | DeepGaze MSDB 前置条件 | 需要 `pixel_per_dva` 观看条件参数（README 已验证，与 technical-design §5"不得从截图猜测"一致 → 缺少显示尺寸/观看距离时不得调用）；CLIP + DINOv2 骨干权重的下载源与体量未验证 | 未验证 |
| U15 | Store 版 Python 3.12 的 venv/pip 行为 | py 启动器可见 3.12（已验证）；Store 版创建 venv 与 pip 安装通常可用（推断），本机未实测；若有缺陷，后备方案（安装用户级官方 3.12.x）触碰"不改全局环境"边界 → 需报总控裁决 | 未验证 |
| U16 | scipy API 漂移对 DeepGaze 示例的影响 | DeepGaze README 示例使用 `scipy.misc.face`，`scipy.misc` 模块在新版 scipy 已移除（公开事实）→ 官方示例在新环境预计直接报错（推断，未运行）；集成时应绕过示例、直接调用 `DeepGazeIIE` API | 未验证 |
| U17 | C 盘空间风险 | C 盘可用 60.0 GB（已验证）；pip/uv/torch/HF 默认缓存均落 C 盘用户目录（通用行为）→ 不执行 §2.2 重定向时，中/大体量下载会蚕食系统盘（风险推断） | 部分验证 |
| U18 | PyPI torch Windows 默认 wheel 是否 CPU-only | 2.14.0 win_amd64 = 124.1 MB（已验证），与历史 CPU 构建同数量级（torch 2.5.1 win_amd64 = 203.0 MB，PyPI 已验证），远小于任何 CUDA 捆绑构建 → 推断为 CPU 构建；未以安装后 `torch.version.cuda is None` 实证，第二波 venv 内第一件事即验证 | 推断（待第二波实证） |

## 6. 第二波可直接照做的执行清单（按序，均为设计，本任务未执行）

1. **C2**：`py -3.12 -m venv projects\game-ui-attention\.venv`；把 §2.2 的缓存重定向段写入 `.venv\Scripts\Activate.ps1` 与 `activate.bat`（venv 在忽略目录内，不进 Git）。
2. **C1**：创建 §3 布局骨架 + `pyproject.toml`（精确 pin：`onnxruntime==1.30.0`、`numpy`、`pillow`；extras `[onnx]/[deepgaze]/[dev]/[eval]`），用 uv 生成 `uv.lock`（或 pip-tools 生成带 hashes 的 requirements.txt）。
3. **C2**：venv 内安装 `[onnx,dev]`（全小体量，无需审批）；运行 `python -c "import torch"` 应失败（未装），`import onnxruntime` 成功并记录 provider 列表。
4. **C2**：下载 `foveacast-v3-3s-fp16.onnx` → `model-cache/foveacast/`，用 release 页 sha256（`842a23f9…`）校验后把期望哈希登记进 `backends/registry.py`。
5. **C2**：实现 `backends/doctor.py` + `onnx_foveacast.py`（运行时读取 ONNX 元数据；记录设备、耗时、峰值内存 → 回填 analysis.json 的 runtime 字段），顺带消解 U4/U5/U6/U8。
6. **C1/C3 并行**：按 §3 所有权推进 contracts/imaging/aoi/metrics/cli（C1）与 overlay/色阶/HTML 报告（C3）；接口签名以总控冻结件为准。
7. **整合（总控串行）**：用 `tests/fixtures/` 合成图跑通 `analyze → summarize → compare` 全链路与退出码 0/2/3/5/6/7 分支 → 作为 G1 工程闭环证据（不含真实截图，不涉密）。
8. **DeepGaze 决策点**：总控确认许可 Issue #15 状态；若可闭合 → 申请中体量下载（§4 #3/#6/#7/#8，报二级总控），在隔离分支先验证 U1/U16（老 API 适配），CPU 延迟不可接受时再报大体量 CUDA wheel（#5，需一级总控批准）。
9. **评估基准（依赖 R2 协议 + 审批）**：UEyes 12.9 GB（#12）需一级总控批准后才下载，落位 `local-data/ueyes/`；未获批前 C1 的 metrics/eval 用合成数据自测。
10. **收口**：全部实测完成后，由总控串行更新 technical-design §3 版本基线与 sources-and-decisions（子任务不改共享方案文档）。

## 附录 A：实测命令与脱敏输出摘录（2026-09-11）

脱敏规则：用户名以 `<user>` 代替；不含机器名；nvidia-smi 进程清单（本机运行应用列表）整段省略；保留盘符、OS 版本、硬件型号。所有命令均为只读查询，未安装/修改/下载任何内容。

### A.1 OS / CPU / RAM / 磁盘

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"
```

```text
Caption        : Microsoft Windows 10 IoT 企业版 LTSC
Version        : 10.0.19044        BuildNumber    : 19044        OSArchitecture : 64 位
Name           : Intel(R) Core(TM) i7-14700KF
NumberOfCores  : 20    NumberOfLogicalProcessors : 28    MaxClockSpeed : 3400
TotalPhysicalMemory  : 68482342912
TotalVisibleMemorySize : 66877288 (KB)    FreePhysicalMemory : 18908704 (KB)
DeviceID  SizeGB   FreeGB
C:        778.3    60.0
D:        1863.0   1024.0
E:        1863.0   324.3
G:        1084.4   877.6
```

### A.2 GPU（nvidia-smi，节选；进程清单已省略——脱敏）

```text
NVIDIA-SMI 596.36        Driver Version: 596.36        CUDA Version: 13.2
GPU 0: NVIDIA GeForce RTX 4070 (WDDM)
       38C   P8   13W / 200W   显存 4787MiB / 12282MiB   GPU-Util 1%
```

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion
```

```text
Name: GameViewer Virtual Display Adapter（虚拟显示适配器，不参与 CUDA 计算）
Name: NVIDIA GeForce RTX 4070   AdapterRAM: 4293918720（32 位字段截断，以 nvidia-smi 12282MiB 为准）
DriverVersion: 32.0.15.9636
```

### A.3 CUDA Toolkit / cuDNN 迹象（只读探测）

```text
where.exe nvcc            → nvcc not found in PATH
$env:CUDA_PATH            → 未设置
C:\Program Files\NVIDIA*  → 仅 "NVIDIA Corporation"（无 CUDA Toolkit 目录）
结论：系统级 CUDA Toolkit / cuDNN 未安装；nvidia-smi 的 "CUDA Version: 13.2" 为驱动支持上限。
```

### A.4 Python / git

```text
py -0p →
  -V:3.14 *   C:\Users\<user>\AppData\Local\Programs\Python\Python314\python.exe
  -V:3.12     C:\Users\<user>\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_…\python.exe
where.exe python →
  C:\Users\<user>\AppData\Local\Programs\Python\Python314\python.exe
  C:\Users\<user>\AppData\Local\Microsoft\WindowsApps\python.exe
python --version → Python 3.14.2
git --version    → git version 2.52.0.windows.1
```

### A.5 仓库状态（只读 git）

```text
git status --porcelain   → （空输出，工作树干净）
git branch --show-current → main
git log --oneline -3 →
  b3ddb20 docs: add wave-1 research status and ownership index
  7cb706c docs: define delegated game UI attention execution plan
  97366d0 docs: initialize game UI attention project plans
```

### A.6 .gitignore 覆盖实测（git check-ignore -v --no-index）

```text
IGNORED  projects/game-ui-attention/.venv/pyvenv.cfg                      <= .gitignore:9  .venv/
IGNORED  projects/game-ui-attention/model-cache/deepgaze2e.pth             <= .gitignore:25 **/model-cache/
IGNORED  projects/game-ui-attention/model-cache/foveacast-v3-3s-fp16.onnx  <= .gitignore:25 **/model-cache/
IGNORED  projects/game-ui-attention/model-cache/umsi++.hdf5                <= .gitignore:25 **/model-cache/
IGNORED  projects/game-ui-attention/runs/run1/manifest.json                <= .gitignore:26 **/runs/
IGNORED  projects/game-ui-attention/local-data/ueyes/sample.png            <= .gitignore:24 **/local-data/
IGNORED  projects/game-ui-attention/work/tmp.txt                           <= .gitignore:27 **/work/
无匹配   projects/game-ui-attention/cache/pip/x.whl                        （裸 cache/ 目录无规则，见 §2.2 注意事项）
```

### A.7 网络来源核查清单（全部核查于 2026-09-11，仅读取页面，未下载资产）

| 来源 | 取得的关键事实 |
| --- | --- |
| https://pypi.org/project/torch/ | 2.14.0（2026-09-02 发布）；Python >=3.10（3.10–3.14 分类器）；win_amd64 wheel 124.1 MB；SPDX 许可表达式 |
| https://pypi.org/project/torch/2.5.1/ | 历史参照：win_amd64 = 203.0 MB（CPU 数量级基准） |
| https://pypi.org/project/torchvision/ | 0.29.0（2026-09-02）；win_amd64 = 1.4 MB；Requires !=3.14.1, >=3.10；BSD |
| https://pypi.org/project/onnxruntime/ | 1.30.0（2026-09-10）；MIT；Python >=3.11（3.11–3.14）；win_amd64 = 14.3–14.7 MB |
| https://download.pytorch.org/whl/ | 根索引存在 cpu、cu126/cu126-full、cu128/cu128-full、cu129、cu130、cu132、cu134 等变体目录 |
| https://github.com/matthias-k/DeepGaze（README 与 releases v1.0.0 资产页） | deepgaze2e.pth = 400 MB；centerbias_mit1003.npy = 8 MB；MSDB 需 pixel_per_dva、用 CLIP+DINOv2；MIT1003 35 ppd / ~1024px 训练口径 |
| https://raw.githubusercontent.com/matthias-k/DeepGaze/main/deepgaze_pytorch/deepgaze2e.py | `DeepGazeIIE(pretrained=True)` → `model_zoo.load_url(…/v1.0.0/deepgaze2e.pth)`；4 骨干 × (3×10) 组件结构 |
| https://raw.githubusercontent.com/matthias-k/DeepGaze/main/deepgaze_pytorch/features/resnext.py | `torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)`（U1/U10 证据） |
| https://github.com/khawkins98/foveacast-training（README 与 releases v0.2.0 资产页） | MIT；UEyes 微调 MSI-Net；v0.2.0 资产：{1s,3s,7s}×{fp16 53.9 MB, int8 30.3 MB} 均带 sha256；240×320 输入、opset 17；Python 3.12 + uv；HF 权重移植需 TF ≈500 MB；作者公布指标口径 |
| https://huggingface.co/alexanderkroner/MSI-Net | TF-Keras SavedModel；MIT；repo ≈100 MB |
| https://zenodo.org/records/8010312 | UEyes_dataset.zip = 12.9 GB（md5 c2d53e6af0a47e1f459416d6839ec2c1）；CC-BY-4.0；62 人 × 1,980 UI × 4 类型 |
| https://github.com/YueJiang-nj/UEyes-CHI2023（README 与 saliency_models/UMSI++/README.md） | 代码含 UMSI++/evaluation/data_processing；UMSI++ 走 conda+TF+Jupyter，需 umsi++.hdf5（分发源未给） |

证据处理说明：期间一次 web_search 返回与 PyPI 实测矛盾的内容（否认 torch 2.14.0 存在），判定为搜索引擎侧不可靠信息，**未采信**；本文全部数值以上表官方页面直接读取结果为准。GitHub REST API 一度限流（403），release 资产体量改用 `releases/expanded_assets/*` 页面核实。