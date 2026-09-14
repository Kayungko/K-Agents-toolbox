# Web 协作校准平台选型与架构设计建议书（A4）

版本：v1（A4 子任务交付）。撰写日期：2026-09-13。
状态：平台选型与架构设计建议书，供二级总控评审；修订需递增版本号并保留旧版可追溯。

---

## 0. 摘要与建议速览（结论先行）

1. **推荐：自研轻量内网 Web 应用**（FastAPI + SQLite/JSONL + 复用并扩展 A1 `index.html` 客户端加 fetch 回传与管理页）。理由：它是**唯一能零改造满足「定时自由观看 → 掩膜 → 凭记忆标记」协议保真度**的方案，且许可最干净（FastAPI=MIT、Flask=BSD-3-Clause、SQLite=public domain）、内网-only 天然可控、与 `consistency.py` 衔接成本趋近于零（回传 JSON 与 A1 结果同构）。
2. **诚实权衡**：Label Studio / CVAT 自托管在「多用户账号、任务管理、审计 UI」上成熟度碾压自研，但两者的标注范式都是**「图始终可见、边看边标」**，原生**无**「定时观看→掩膜→凭记忆作答」流程；要保真必须重写标注前端，此时其内置 UI 价值大部分被抵消、而服务端/数据库/导入导出开销与坐标单位转换成本仍保留（Label Studio 导出坐标为**百分比**、需换算，见 §3.2）。因此在本研究线的硬约束下，自研的综合成本反而更低、保真度最高。
3. **红线全覆盖**：方案内网 bind（`127.0.0.1` 或内网接口）+ 不可猜 session token + admin token；截图仅存 `local-data/`（Git 忽略）；禁任何公有云 SaaS；新增设计稿=运行期热加（不改服务、不重出包）。
4. **降级路径**：服务机不可用时**回退 A1 静态文件夹模式**（`index.html` 与静态包已存在，同事本地打开+下载 JSON 交回），数据口径完全一致。

---

## 1. 目标与范围

### 1.1 需求溯源（用户原话要点）

现有 A1 内部粗标注校准工具（`calibration/`：`protocol.md` + `make_study_package.py` + 纯客户端 `index.html` + `consistency.py`）分发方式=拷贝文件夹、回收方式=同事下载 JSON 交回。用户提出三点升级：

1. 同事协助标记环节要一个 **web 端**，方便分发协作；
2. 后台要能**并行随时新增 UI 界面设计稿**（不等重新出包）；
3. 标记信息要能**回传**到服务端自动收集。

本建议书产出：平台选型对照 + 推荐架构草图 + responses schema 草案 + config 哈希版本化点 + 风险缓解。

### 1.2 既有协议流程（本建议书的地基，已读源码核实）

- 每图：注视十字 500 ms → 图片展示 N 秒（默认 3，可配 5，进度条）→ 掩膜遮挡 → **凭记忆**作答（点选 1 个第一眼落点 + 拖 1–3 个主看框）；
- 结果 JSON 含：`schema_version` / `study_id` / `participant_id`（匿名自报）/ `started_at_utc` / `finished_at_utc` / `duration_seconds` / `shuffle_seed` / `user_agent` / `completed` / `images[]`（每图 `index/filename/sha256/width/height/screen_type/display_started_at_utc/display_ended_at_utc/display_ms/responded/first_look{x,y}/boxes[]{x,y,width,height}`）；
- `make_study_package.py` 的 `study_id = sha256(排序后(sha256,source_name) 集合 + 时长)[:16]`；`config.json` 记录图清单+sha256+顺序种子+时长；
- `consistency.py` 读 `config.json` + 回收 responses JSON + 模型 `analysis.json`/`density.npy`（按 `input.image_sha256` 对齐），消费 response 顶层 `participant_id/completed/images[]` 与每图 `sha256/responded/first_look/boxes`。

### 1.3 口径对齐（呼应先例文档）

- 与 [diy-eyetracker-research.md](diy-eyetracker-research.md) §5 / [benchmark-protocol.md](../research/benchmark-protocol.md) §8.1 的 **config 哈希版本化纪律**对齐：任何影响可比性的参数（图集合/协议版本/客户端版本/服务端版本/时长/坐标口径）必须版本化入 config 哈希，任一项不同 → 分表并列，不得同表比较。
- 与 [calibration/protocol.md](../calibration/protocol.md) §0.1/§8 对齐：本文档可公开，不含截图本体、不含本机路径、不含凭据或身份信息。

---

## 2. 硬约束（红线，任何方案不得违反）

| # | 红线 | 本建议书处置 |
| --- | --- | --- |
| R1 | SGAME 截图与标注数据仅限内网/本机；禁公有云 SaaS；禁上传外部 | 所有方案仅评估自托管；bind 内网/localhost；禁外链 |
| R2 | 协议保真度硬需求：定时观看→掩膜→凭记忆标记，不能被「边看边标」替代 | 作为对照表独立列，作为推荐否决项（§4） |
| R3 | 结果数据含 sha256/展示时长/第一眼坐标/主看框/participant id/时间戳；须被 consistency.py 直接消费或给适配层 | §5.4/§5.6 |
| R4 | 后台新增设计稿=运行期热加载（不重启、不重出包），新图自动纳入后续参与者图序（随机化种子按参与者会话记录） | §5.3/§5.5 |
| R5 | 许可纪律：Apache-2.0/MIT 优先；GPL 仅服务端运行不分发时评估 | §3 各方案许可逐项核实 |
| R6 | 自研代码入 `calibration/` 或新目录由 L2 定（本文只写建议书） | §5.2 备注 |

---

## 3. 候选方案盘点

### 3.1 方案 a：自研轻量内网 Web 应用（推荐）

**形态**：Python **FastAPI**（或 Flask）+ **SQLite**（图库/会话元数据）+ **JSONL**（responses append-only）+ 复用扩展 A1 `index.html`（加 `fetch` POST 回传 + 会话 config 拉取）→ 内网机器自托管。

**许可（已核实）**：FastAPI = MIT（GitHub API 一手抓取，`license.key=mit`）；Flask = BSD-3-Clause（`pallets/flask`，`license.key=bsd-3-clause`）；SQLite 为 public domain；均无 copyleft 义务，符合红线 R5。

**端点设计（草案）**：

| 方法 | 路径 | 职责 | 访问控制 |
| --- | --- | --- | --- |
| GET | `/s/{session_id}` | 参与者标注页（扩展版 `index.html`，注入该会话 config） | 不可猜 `session_id` token |
| GET | `/api/session/{session_id}` | 返回该会话 `config`（图清单+sha256+**按会话种子**的图序+时长+config_hash） | 同上 |
| GET | `/images/{filename}` | 静态图服务（从 `local-data/` 图片库只读服务） | 同上（仅本服务内引用） |
| POST | `/api/responses` | 接收回传结果，校验后 append 到 JSONL，返回回执 id | 会话级写 |
| POST | `/admin/tasks` | **上传新增设计稿**（multipart 图）+ 登记 sha256/尺寸/screen_type | admin token |
| GET | `/admin/tasks` | 列出图库（sha256/尺寸/类型/active/added_at） | admin token |
| POST | `/admin/tasks/{sha256}/retire` | **下架**（soft-delete，置 active=false，不物理删除以保 sha256 引用完整） | admin token |
| GET | `/admin/responses` | 导出 responses（JSONL 或按 participant 拆分） | admin token |
| DELETE | `/admin/participants/{participant_id}` | 按 participant id 删除其全部行（数据删除权） | admin token |
| GET | `/admin/sessions` | 会话列表（匿名码、图序种子、状态） | admin token |

**并发与备份**：responses 用 **JSONL append-only**（每条自包含、`fsync` 后追加，天然抗部分写、无锁竞争）；图库/会话用 **SQLite WAL 模式**（多读单写、崩溃可恢复）。备份=定时 `rsync`/拷贝 `local-data/web-calibration/` 整个目录（JSONL+SQLite 均单文件，易复制）。

**participant 匿名码**：参与者自报匿名 id（同 A1）；服务端另发**不可猜 `session_id`**（`secrets.token_urlsafe`），二者分离——匿名码用于分组/删除权，session_id 用于访问控制与回传绑定。

**内网 bind 与访问控制**：`uvicorn --host 127.0.0.1` 或内网接口；admin 端点 `X-Admin-Token` 头；禁外网路由（应用零外链、零外部 fetch、零 data:URI 大图，延续 A1 断言）；访问日志脱敏记录。

**工时估算**：后端端点+存储 ~2–4 人日；客户端扩展（fetch 回传+会话 config 拉取）~0.5–1 人日；管理页（上传/列表/下架/导出/删除）~1 人日；consistency 适配/导出 ~0.5 人日；测试（协议流程+并发写+删除权+降级）~1 人日。**合计约 5–8 人日（单人）**，且大量代码（标注前端逻辑）零新写、直接复用 A1。

### 3.2 方案 b：Label Studio 自托管

**许可（已核实）**：Apache-2.0（`HumanSignal/label-studio` GitHub API，`license.key=apache-2.0`，约 28k stars，近端仍活跃）。符合红线 R5。

**部署形态（已核实）**：`pip install label-studio` 或 Docker（`heartexlabs/label-studio`）；默认 `localhost:8080`；后端 DB 用 SQLite/PostgreSQL；支持多用户账号、项目、任务锁定（防双开）。

**动态加 task（已核实+注意）**：支持 JSON 批量导入与 REST API 加 task；但官方 tasks 文档明示「避免频繁导入，每次导入需较长后台操作，约每 30 秒一次才无过载」——即 LS 的「加 task」是**批量后台导入**，非为「随时热加单张」设计（对需求②是体验降级点）。

**结果导出 schema（已核实）**：raw JSON 中 annotation 的 `result[]` 含 `type/from_name/to_name/value`；**关键差异：图像标注坐标用「占整图百分比」而非像素**（导出文档原文：`Image annotations exported in JSON format use percentages of overall image size, not pixels`）。导出端点 `GET /api/projects/{id}/export?exportType=JSON`；Community 版导出为同步（大项目有 90s 超时风险，本线规模 5–10 人×~20 图不构成问题）。

**协议保真度（核心否决点，已核实+推断）**：
- 标注界面由**声明式 XML labeling config** 驱动（`<Image value="$image">`、`<Rectangle>`、`<KeyPoint>`、`<Choices>` 等 tag），**图片在标注时始终可见**（`<Image>` tag 即把图渲染出来供边看边画）——这与「掩膜后凭记忆作答」直接冲突；
- 官方文档**未提供**任何「定时展示→自动掩膜→记忆作答」的 tag 或流程；
- Frontend 库（LSF）自 1.11.0 起**已弃用为独立分发**，其事件系统（`labelStudioLoad`/`beforeSaveAnnotation` 等）仅用于回调/监听，**非**用于控制图片可见时长的时序流程；
- 结论（推断）：要保真，需在 LS 之外**自写前端**实现定时观看+掩膜，等价于重写 A1 客户端，LS 的内置标注 UI 价值被抵消，而其服务端/数据库/导入导出/百分比坐标换算成本仍保留。

**内网部署复杂度**：中（单进程即可，但账号/项目/token 管理有学习成本）。**工时**：部署 0.5–1 天 + 范式改造（≈自研客户端工时）+ 坐标换算适配器 + 导入/导出脚本，综合**不低于甚至高于自研**。

### 3.3 方案 c：CVAT 自托管

**许可（已核实）**：MIT（`cvat-ai/cvat` GitHub API，`license.key=mit`，约 16.7k stars，近端活跃）。符合红线 R5。

**部署形态（已核实）**：Docker Compose 为主；Python 后端 + 前端 SPA；REST API（Token 认证）支持 `POST /api/tasks` 建任务、`GET /api/tasks/{id}/annotations?format=...` 导出（CVAT XML/COCO/YOLO/VOC 等）；多用户与角色。

**范式匹配度（已核实+推断）**：CVAT 定位是「图像/视频/3D 标注」的形状标注工具（bbox/polygon/keypoint 直接画在**始终可见**的图/帧上），支持视频逐帧与属性，但**无**「定时观看→掩膜→凭记忆」流程。与协议保真度冲突同 Label Studio；相对优势是导出坐标多用**像素**（与自研 schema 更近）而非百分比，且 MIT 许可最宽松；劣势是部署更重（多容器）、面向 CV 数据集的语义与本研究「主观回忆式粗标注」差距更大。

**工时**：部署 1 天左右 + 同等的范式改造 + 导出格式（COCO/CVAT XML）→ 自研 schema 适配。综合成本高、保真度仍达不到。

### 3.4 方案 d：现状基线（A1 文件夹分发 + JSON 回收）

**形态**：`make_study_package.py` 出静态包 → 拷贝/共享目录分发 → 同事本地打开 `index.html` → 下载 JSON 交回 → `consistency.py` 打分。

**对照意义**：这是**降级路径与数据口径基准**，非「满足升级需求」的候选。三点升级需求（web 分发/后台热加稿/自动回传）**全部不满足**；但它的标注客户端逻辑是方案 a 的直接复用对象，且协议保真度 100%（就是协议本体）。工时=0（已交付）。

### 3.5 方案 e：组织内既有协作工具（飞书表单/多维表格）

**形态**：飞书表单/多维表格（Base）做分发与回传。

**结论（推断，大概率不能承载）**：Base 字段类型为文本/数字/单选/多选/日期/附件/人员等，**无「图片上点选坐标 + 拖拽画框」的交互控件**；表单无法在图片上采集任意像素坐标与 1–3 个框，更无法强制「先定时展示、后掩膜、凭记忆作答」的时序流程。可承载「匿名码收集+附件交回 JSON」这一层（相当于把「交回」从聊天/邮件换成表单附件），但**无法承载协议核心的标注交互**，因此不能作为标注平台，最多作为「降级交回通道」的补充。

---

## 4. 五方案对照表

| 维度 | a 自研轻量内网 | b Label Studio 自托管 | c CVAT 自托管 | d A1 基线 | e 飞书表单/多维表格 |
| --- | --- | --- | --- | --- | --- |
| 许可 | FastAPI=MIT / Flask=BSD-3 / SQLite=PD（已核实） | Apache-2.0（已核实） | MIT（已核实） | 自研 | 闭源 SaaS（内网数据外置风险） |
| 部署形态 | 单进程 FastAPI + 单文件存储 | pip 或 Docker，单进程/DB | Docker Compose 多容器 | 纯静态文件夹 | 无部署 |
| 多用户/账号 | 无账号体系（匿名码+会话 token），够用 | **成熟账号/项目/角色/任务锁** | 成熟账号/角色 | 无 | 有（但无标注控件） |
| 后台热加稿 API | `POST /admin/tasks` 运行期热加，立即入图序（自建） | 有 API 但为**批量导入**、频繁导入被官方不建议 | 有 API（task 导入） | 无（需重出包） | 无 |
| 结果回传与导出 schema | 直接回传，JSONL；导出=逐份 A1 同构 JSON | 有导出，但坐标为**百分比**需换算、含 LS 专有字段 | 有导出（COCO/XML 等），需转自研 schema | 手动交回 JSON | 附件交回 JSON |
| 协议保真度（定时观看+掩膜+记忆标记） | **原生 100%（复用 A1 客户端）** | **原生不支持**，需重写前端（推断） | **原生不支持**（推断） | 100%（即协议本体） | **不支持**（推断） |
| 内网-only 可行性 | 高（bind+token 自控） | 中（自托管可，但需配置账号/反代） | 中（自托管可） | 天然内网 | **否（数据入飞书云，违红线 R1）** |
| 匿名化 | 匿名码自报+不收集账号，自控 | 需禁用/绕过账号实名（社区版账号为本地用户） | 需管理用户（默认需账号） | 匿名码自报 | 表单可匿名，但数据上云 |
| 与 consistency.py 衔接成本 | **≈0（同构）+ 可选导出适配** | 需写坐标换算（百分比→像素）+字段映射 | 需写格式转换 | 0 | 0（仅交回层） |
| 搭建工时 | 5–8 人日 | 部署 0.5–1 天 + 范式改造（≈自研客户端） | 部署 1 天 + 范式改造 + 格式转换 | 0（已交付） | 0 开发但不可用 |
| 维护负担 | 低（自研可控，依赖少） | 中（版本升级、DB、账号） | 中高（多容器、版本） | 极低 | 低 |
| 单点故障（服务机宕机） | 有，但**可降级回 d 方案** | 有，降级需另备静态包 | 有 | 无（分散本地） | 飞书侧可用但违红线 |

---

## 5. 推荐结论与架构草图

### 5.1 推荐结论与诚实权衡

**推荐方案 a（自研轻量内网 Web 应用）**。核心理由是**协议保真度**：本研究线价值在于「粗标注—模型一致性」的**范式严谨性**，定时观看→掩膜→凭记忆是该范式的关键，任何「边看边标」的通用标注工具都会**系统性污染数据**（回忆式 vs 边看边标是两种不同的认知任务）。自研方案直接复用 A1 已验证的客户端逻辑，保真度=协议本体；Label Studio/CVAT 要保真则必须重写前端，等于「付了平台的开销、又写了客户端」。

**诚实权衡（Label Studio 的真正优势与本线的不匹配）**：
- LS 的成熟多用户账号、任务锁定、标注进度看板、审计 UI 是自研方案**短期做不出的**；若未来需要「多标注员同图交叉标注 + 一致性仲裁 + 进度管理」这类规模化协作，LS 的增量价值会显著上升，届时值得**重新评估**（尤其若协议放宽为通用标注范式）。
- 但本线硬约束（5–10 人、匿名自报、协议保真、内网-only、schema 与 consistency 对齐）下，LS 的优势项大多**用不上或被范式改造抵消**，而劣势项（百分比坐标换算、批量导入、账号体系、升级维护）都实打实要付。
- 结论：**短期选 a；中期若协作规模扩大且允许放宽范式，再评估迁移 b**（迁移成本=写坐标换算适配器，schema 侧已预留 config_hash 分表能力）。

### 5.2 模块图（文字）

```text
[管理员] --上传/列表/下架--> [admin API (token)] ----+
                                                       v
                                      ┌────────────────────────────┐
                                      │ 图片库 local-data/web-images │
                                      │ （Git 忽略；sha256 登记表）   │
                                      └──────────────┬─────────────┘
                                                     │ 读图
[参与者浏览器] --GET /s/{session_id}--> [会话发号器] ─┴─> [标注客户端 index.html(扩展A1)]
        ^                                    │  匿名码 + 图序种子(按会话) + config_hash
        │                                    │
        └── POST /api/responses <────── fetch 回传 ──┘
                                                     │
                                                     v
                                      ┌────────────────────────────┐
                                      │ responses.jsonl (append-only)│
                                      │ 版本化 schema；每行=一份结果  │
                                      └──────────────┬─────────────┘
                                                     │ 导出/适配
                                                     v
                                      ┌────────────────────────────┐
                                      │ consistency.py（消费 A1 同构）│
                                      └────────────────────────────┘
```

自研代码建议入 `calibration/web/`（新目录，由 L2 定）；截图本体仍仅入 `local-data/`（沿用 Git 忽略，与 `calibration/tools/` 只放代码的既有纪律一致）。

### 5.3 端点设计（完整，见 §3.1 表；此处补关键语义）

- **运行期热加稿**：`POST /admin/tasks` 上传后即时计算 sha256（与 `imaging.sha256_file` 同口径）、读尺寸、登记 `active=true`；**不重启、不重出包**。后续新会话发号时，图库=「当前 active 图」按会话种子 shuffle；已进行中的会话图序**不回溯变更**（快照式发号，保证已发会话的可复现性），仅新会话纳入新图。
- **图序种子按会话记录**：会话发号时 `seed_i = 随机（或全局 seed，admin 可配）`，写入 session 记录与 `config`；response 回传时带 `shuffle_seed`，`consistency.py` 可据此复现图序（呼应 protocol §3）。
- **回传校验**：服务端对回传做最小校验——`schema_version` 匹配、`sha256 ∈ 当前/历史图库`、坐标钳制、`completed` 标志一致；非法行记录不丢弃（标记 `_server.accepted=false`）供排查。
- **下架语义**：`retire` 置 `active=false` 而非物理删除，保证历史 response 的 sha256 引用始终可解析（protocol §7.4 错图绑定校验依赖 sha256 对得上图库）。

### 5.4 responses schema 草案（存储行字段）

回传体**保持与 A1 `window.__result` 完全同构**（`consistency.py` 直接消费），服务端附加 `_server` 信封。JSONL 每行：

```jsonc
{
  // ---- 与 A1 同构（consistency.py 直接消费）----
  "schema_version": "game-ui-attention-calibration-response/v1",
  "study_id": "…",                      // 图集合+时长合成哈希（沿用 make_study_package）
  "participant_id": "p01",              // 匿名自报
  "started_at_utc": "2026-09-13T…Z",
  "finished_at_utc": "2026-09-13T…Z",
  "duration_seconds": 3,
  "shuffle_seed": 123456,               // 按会话记录
  "user_agent": "…",
  "completed": true,
  "images": [
    {
      "index": 0,
      "filename": "img_000.png",
      "sha256": "…",                    // 图 sha256（红线 R3）
      "width": 1920, "height": 1080,
      "screen_type": "…",
      "display_started_at_utc": "…",
      "display_ended_at_utc": "…",
      "display_ms": 3000,               // 展示时长（红线 R3）
      "responded": true,
      "first_look": { "x": 960.0, "y": 300.0 },     // 第一眼坐标（红线 R3）
      "boxes": [ { "x": 100.0, "y": 200.0, "width": 400.0, "height": 300.0 } ]
    }
  ],

  // ---- 服务端信封（consistency.py 忽略未知键，故零侵入）----
  "_server": {
    "session_id": "…",                  // 不可猜会话 token（访问控制）
    "received_at_utc": "2026-09-13T…Z", // 服务端时间戳（红线 R3）
    "config_hash": "…",                 // §5.5
    "client_version": "…",              // index.html 代码哈希
    "server_version": "…",              // app 代码哈希
    "device": null,                     // 预留：未来眼动数据（"tobii-fusion-250" 等）
    "device_params": null,              // 预留：未来聚类/标定参数（版本化入 config_hash）
    "remote_addr_redacted": true,       // 不落 IP 原文
    "accepted": true                    // 校验通过标记
  }
}
```

### 5.5 config 哈希版本化点（呼应 diy §5 / benchmark §8.1 纪律）

定义 `config_hash = sha256(canonical_json({...}))`，参与项：

1. **图集合版本**：`active` 图库排序后 `sha256` 列表（= 图集合内容指纹）；
2. **协议版本**：`protocol.md` 版本 + `schema_version`；
3. **客户端版本**：`index.html` 扩展版代码内容哈希（Git 提交或文件 sha256）；
4. **服务端版本**：`calibration/web/` 应用代码哈希；
5. **观看时长**：`duration_seconds`；
6. **图序种子策略**：每会话 `shuffle_seed`（写入 response，见 §5.3）；
7. **（预留）设备参数**：`device` / `device_params`（未来眼动，按 L2 已冻结的「适配器参数版本化入 config_hash」纪律，diy §9.2）。

**可比性规则**：`config_hash` 任一参与项不同 → 该批结果与旧批**分表并列、显式标注差异项**，不得同表比较（对齐 benchmark §8.1）。与既有 `study_id`（图集合+时长）的关系：`study_id` 是「批次身份」，`config_hash` 是「全量可比性指纹」，后者是前者的超集。

### 5.6 与 consistency.py 的衔接（适配层设计）

- **零侵入路径**：response 体与 A1 同构，`_server` 为 `consistency.py` 不读的未知键，故**理论上可零改动消费**；仅需把 JSONL 逐行拆成「每 participant 一个 `.json`」的目录形态（`load_responses` 已支持目录扫描）。
- **推荐显式适配**：新增极薄的 `calibration/web/export_responses.py`：读 `responses.jsonl` → 校验 `_server.accepted` → 按 `participant_id` 拆分写回 A1 同构 JSON（剥除 `_server` 或保留均可）→ `consistency.py --responses <导出目录>` 原样运行。
- **删除权重跑**：`DELETE /admin/participants/{id}` 删除该 participant 的 JSONL 行后，`export_responses.py` 重新导出即自动剔除该人，`consistency.py` 重跑即得删后报告（满足 protocol 已有删除权条款的 web 化实现）。

### 5.7 工时估算（汇总）

| 方案 | 搭建工时 | 备注 |
| --- | --- | --- |
| a 自研 | **5–8 人日（单人）** | 客户端逻辑零新写 |
| b Label Studio | 0.5–1 天部署 + ≈自研客户端工时 + 坐标换算/导入导出 | 保真改造抵消平台收益 |
| c CVAT | 1 天部署 + 范式改造 + 格式转换 | 多容器维护更重 |
| d A1 | 0 | 不满足升级需求 |
| e 飞书 | 0 开发 | 不可承载标注交互且违红线 R1 |

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 服务机可用性（同事访问时段需开机） | 服务机宕机/下班关机→同事无法访问 | ①约定访问时段开机；②**降级路径：回退 A1 静态文件夹模式**（`make_study_package.py` 出包→共享目录→本地打开→下载 JSON 交回，口径一致）；③服务用 `uvicorn` 单进程、秒级冷启 |
| 并发写冲突 | 多人同时 POST responses | JSONL **append-only**（单行追加+`fsync`，天然免锁）；图库/会话用 SQLite **WAL**；并发量（5–10 人）远低于任何锁竞争阈值 |
| 截图保密 | 截图外泄 | 内网 bind（127.0.0.1/内网接口）+ 会话 token + admin token；应用**零外链/零外发**；禁外网路由；访问日志脱敏（不落 IP 原文）；截图仅存 `local-data/`（Git 忽略） |
| 参与者匿名与删除权 | 身份泄露/无法撤回 | 只收集匿名码，不收集姓名工号账号；`DELETE /admin/participants/{id}` 删行 + 重导出 + `consistency.py` 重跑（protocol 已有删除权条款） |
| schema 演进 | 新旧结果混表 | responses `schema_version` + `config_hash` 版本化；任一项变更→分表（§5.5） |
| 未来眼动数据接入 | 现 schema 无法承载眼动 | 预留 `device`/`device_params` 字段 + 版本化入 config_hash；同平台扩展为「人侧真值设备无关」的统一回传层 |
| 依赖腐烂（自研轻量） | FastAPI/SQLite 停更 | 依赖极简且均为最主流项目；锁定版本；服务端不分发则无 GPL 顾虑（本栈无 GPL 组件） |

---

## 7. 来源清单

### 7.1 外部 web 来源（访问日期均为 2026-09-13）

| # | 来源 URL | 取得的关键事实 | 状态 |
| --- | --- | --- | --- |
| 1 | https://api.github.com/repos/HumanSignal/label-studio | Label Studio：**Apache-2.0**、约 28.2k stars、TypeScript、默认分支 develop、近端活跃 | 已核实 |
| 2 | https://labelstud.io/guide/install.html | pip/Docker 自托管安装、`localhost:8080`、数据卷持久化 | 已核实（经 web_search 汇总 + 官方导航确认） |
| 3 | https://labelstud.io/guide/labeling.html | 标注界面由 XML labeling config 驱动；`<Image value="$image">` 图始终可见边看边画；区域类型 Rectangle/Ellipse/KeyPoint/Polygon/Brush；任务锁定防双开 | 已核实 |
| 4 | https://labelstud.io/guide/tasks.html | JSON 批量导入 task；「避免频繁导入，每次导入需较长后台操作，约每 30 秒一次才无过载」 | 已核实 |
| 5 | https://labelstud.io/guide/export.html | 导出端点 `GET /api/projects/{id}/export?exportType=JSON`；**图像标注 JSON 用占整图百分比而非像素**；raw JSON 结构（result[] 的 type/from_name/to_name/value）；Community 版同步导出 90s 超时风险 | 已核实 |
| 6 | https://labelstud.io/guide/frontend_reference.html | LSF 自 1.11.0 起弃用为独立分发；事件系统仅回调/监听（labelStudioLoad/beforeSaveAnnotation 等），非时序流程控制 | 已核实 |
| 7 | https://api.github.com/repos/cvat-ai/cvat | CVAT：**MIT**、约 16.7k stars、Python、图像/视频/3D 标注定位、近端活跃 | 已核实 |
| 8 | https://docs.cvat.ai/docs/api_sdk/api/ | REST API Token 认证；`POST /api/tasks` 建任务、`GET /api/tasks/{id}/annotations?format=...` 导出（CVAT XML/COCO/YOLO/VOC 等） | 已核实（经 web_search 汇总） |
| 9 | https://api.github.com/repos/fastapi/fastapi | FastAPI：**MIT**、约 102k stars、Python、近端活跃 | 已核实 |
| 10 | https://api.github.com/repos/pallets/flask | Flask：**BSD-3-Clause**、约 74.7k stars、近端活跃 | 已核实 |

### 7.2 项目内既有文档引用（非外部来源，仅作口径引用）

- [calibration/protocol.md](../calibration/protocol.md)（§0–§9：协议流程、图序种子、匿名化、排除标准、数据流向）
- [calibration/tools/index.html](../calibration/tools/index.html)（客户端流程与 `window.__result` schema）
- [calibration/tools/consistency.py](../calibration/tools/consistency.py)（消费字段与排除/打分逻辑）
- [calibration/tools/make_study_package.py](../calibration/tools/make_study_package.py)（`study_id` 合成、config schema、sha256 口径）
- [docs/proposals/diy-eyetracker-research.md](diy-eyetracker-research.md)（§5 config 哈希版本化纪律、§9.2 L2 冻结「适配器参数版本化入 config_hash」）
- [docs/research/benchmark-protocol.md](../research/benchmark-protocol.md)（§8.1 两结果同表比较充要清单）

---

## 8. 证据分级汇总

- **已核实（2026-09-13 一手抓取）**：Label Studio 许可 Apache-2.0 与部署形态、XML labeling config 的「图边看边标」范式、JSON 批量导入的「避免频繁导入」限制、导出 JSON 的**百分比坐标**与 raw schema、LSF 自 1.11.0 弃用与事件系统性质；CVAT 许可 MIT 与图像/视频/3D 定位；FastAPI MIT、Flask BSD-3-Clause。
- **推断**：Label Studio / CVAT **原生无**「定时观看→掩膜→凭记忆」流程（由其「图始终可见 + 声明式 config」证据合理推导，未实测）；飞书表单/多维表格无法承载「图上点坐标+拖框」交互（据 Base 字段类型与表单控件能力推导，未实测）；各方案工时区间估计；LS「要保真需重写前端」的工作量等价性。
- **未核实**：Label Studio/CVAT 在自改前端后的实际工程难度与精确工时（需实做验证）；CVAT 导出的精确坐标单位（推断为像素，未取导出样例原文）；飞书表单的精确控件清单（未逐项取证）。

---

## 9. L2 裁决记录（2026-09-13，二级总控）

1. **推荐采纳**：方案 a（自研轻量内网 Web 应用）为选定路线；Label Studio 迁移仅作中期预案（协作规模扩大且允许放宽范式时重新评估），短期不启动。否决依据认可：协议保真度（定时观看→掩膜→凭记忆）为硬约束，LS/CVAT 的"边看边标"范式改造成本抵消平台收益（官方文档证据链完整）。
2. **代码落位**：自研代码入 **`calibration/web/`**（新子目录，仅代码与模板）；运行期数据（图片库/SQLite/JSONL/导出）一律落 `local-data/`（Git 忽略），与 calibration/tools 只放代码的既有纪律一致。
3. **图序种子策略**：默认**每会话独立种子**（admin 可配全局种子）；种子写入 session 记录与 response `shuffle_seed`，保证已发会话图序可复现、新加稿仅影响新会话（快照式发号语义认可）。
4. **飞书结论标注维持**："不能承载图上点坐标+拖框交互"为推断（未逐项取证），仅作降级交回通道备选；不作为标注平台。
5. **实现前置条件**：① 用户/组织批准启动实现线（新内网服务属新增运行态，需明示）；② 第三方包（fastapi/uvicorn/python-multipart 等）由 C2 线按既有纪律安装入共享 venv（唯一包维护者），实现线不得自行 pip install；③ 实现验收须含：协议流程 E2E（定时→掩膜→回传）、并发写、删除权重跑、降级回退演练、零外链断言复验。

---

*本文档为 A4 子任务独占产出（单文件）；不修改任何其他文件、不起任何服务、不下载、不写 Git。*
