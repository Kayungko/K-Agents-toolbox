# Web 协作校准平台（calibration/web）

内网自托管轻量 Web 应用，把 A1「拷贝文件夹 → 下载 JSON 交回」升级为「发链接 → 浏览器标注 → fetch 自动回传」，
同时支持后台**运行期热加设计稿**（不重启、不重出包）与**删除权**。权威设计 = `docs/proposals/web-calibration-platform-research.md`（A4，§9 L2 裁决采纳）。

- 后端：FastAPI + SQLite（WAL）+ responses JSONL（append-only）；
- 客户端：复用并扩展 A1 `index.html`（定时观看 → 掩膜 → 凭记忆标注，逐字保留流程，仅加 fetch 拉会话 config + fetch 回传）；
- 数据：`local-data/web-images/`（图片本体）+ `local-data/web-calibration/`（SQLite/JSONL/access.log/export），均为 Git 忽略目录。

> **红线**：本平台**仅内网**使用。截图与标注数据禁出内网；serve 的 HTML 零外链；访问日志脱敏（不落 IP 原文）。

---

## 0. 文件清单

| 文件 | 职责 |
| --- | --- |
| `app.py` | FastAPI 应用（端点全集 + admin token + 脱敏日志 + 启动入口） |
| `store.py` | 存储层（SQLite WAL + JSONL append-only + 快照式发号 + 删除权） |
| `hashing.py` | config_hash（七参与项）/ study_id / 代码哈希（纯 stdlib） |
| `index.html` | 参与者标注页（扩展 A1，fetch 拉 config + fetch 回传） |
| `admin.html` | 管理页（上传/列表/下架/导出/删除/会话） |
| `export_responses.py` | 导出 CLI：JSONL → A1 同构 JSON（按 participant 拆分） |

---

## 1. 启动

依赖（由 C2 线安装入共享 venv）：`fastapi` / `uvicorn` / `python-multipart` / `httpx`（测试）。运行用共享 venv 的 python：

```powershell
# 项目根
cd D:\git\K-Agents-toolbox\projects\game-ui-attention
.\.venv\Scripts\python.exe calibration\web\app.py --host 127.0.0.1 --port 8000
```

启动时控制台打印 **admin token**（若未通过 `--admin-token` 或环境变量 `CALIB_ADMIN_TOKEN` 提供，则随机生成）。
内网供同事访问时改 `--host` 为本机局域网 IP（如 `--host 192.168.x.x`），端口按需放开（仅内网防火墙）。

其他参数：

```text
--data-dir      SQLite/JSONL 数据目录（默认 local-data/web-calibration）
--images-dir    图片库目录（默认 local-data/web-images）
--duration      每图观看秒数（默认 3，可配 5）
--seed          全局图序种子（默认缺省 = 每会话独立随机种子）
--admin-token   显式指定 admin token（默认读 CALIB_ADMIN_TOKEN，均缺省则随机生成打印）
```

> admin token 仅打印到控制台；**不写文件、不入日志原文、不入仓库**。忘记时重启服务即可重新生成（旧 token 失效）。

---

## 2. 管理员操作（管理页 `http://127.0.0.1:8000/admin`）

管理页本身为静态 HTML，所有 `/admin/*` 接口需 `X-Admin-Token` 请求头。在页面顶部粘贴 token 后：

1. **上传设计稿**：选择图片（png/jpg/jpeg/bmp/tif/tiff/webp）+ 填 `screen_type` → 上传即登记 `active=true`，**立即纳入后续新会话图序**（不重启、不重出包）。
2. **图库列表**：查看 sha256/尺寸/类型/状态；**下架**（soft-delete，置 `active=false`，不物理删除，历史 sha256 引用仍可解析）。
3. **会话列表**：查看已发会话；**复制标注链接**（`/s/<session_id>`）。
4. **导出 responses**：下载全部行（含 `_server` 信封）；按 participant 拆分的 A1 同构导出用 `export_responses.py`。
5. **删除参与者**：输入 `participant_id` → 删除该人全部 JSONL 行（删除权）。

等价 curl（供脚本化）：

```powershell
# 上传（multipart）
curl -X POST -H "X-Admin-Token: <token>" -F "file=@设计稿.png" -F "screen_type=reward" http://127.0.0.1:8000/admin/tasks
# 图库
curl -H "X-Admin-Token: <token>" http://127.0.0.1:8000/admin/tasks
# 下架
curl -X POST -H "X-Admin-Token: <token>" http://127.0.0.1:8000/admin/tasks/<sha256>/retire
# 删除参与者
curl -X DELETE -H "X-Admin-Token: <token>" http://127.0.0.1:8000/admin/participants/p01
# 会话列表
curl -H "X-Admin-Token: <token>" http://127.0.0.1:8000/admin/sessions
```

---

## 3. 发号与回传

会话通过**不可猜 session_id**（`secrets.token_urlsafe`）访问控制：

- 发号：目前由管理员按需创建会话。测试/脚本可用一段最小 Python 调用 `store.Storage.create_session()` 拿到 session_id 与标注链接；
  图序按**会话种子** shuffle（默认每会话独立种子，`--seed` 配全局种子），**快照式**冻结图清单——进行中会话不受后续加稿/下架影响。
- 参与者打开 `/s/<session_id>` → 页面 fetch 拉会话 config → 完成标注 → fetch POST `/api/responses?session_id=<id>` 自动回传。

回传体与 A1 `window.__result` 完全同构；服务端校验后 append 到 JSONL，返回回执 id。非法行**不丢弃**，标记 `_server.accepted=false` 落盘供排查。

---

## 4. 导出 → consistency.py 打分

```powershell
cd D:\git\K-Agents-toolbox\projects\game-ui-attention

# 1) 导出（过滤 accepted、按 participant 拆分、剥 _server）
.\.venv\Scripts\python.exe calibration\web\export_responses.py `
    --responses local-data\web-calibration\responses.jsonl `
    --out local-data\web-calibration\export

# 2) 一致性打分（consistency.py 直接消费导出目录；--runs 为已跑 analyze 的 run 目录）
.\.venv\Scripts\python.exe calibration\tools\consistency.py `
    --package-config <对应研究包>/config.json `
    --responses local-data\web-calibration\export `
    --runs <runs-dir> `
    --out local-data\calibration-outputs\<id>
```

> `--package-config` 的 config.json 图清单需与 web 图库一致（同一批截图）。web 侧 `study_id` 与 A1 生成器同口径
> （图集合 + 时长合成哈希），`config_hash` 是其「全量可比性指纹」超集。

**删除权重跑**：`DELETE /admin/participants/{id}` 删行 → 重新运行 `export_responses.py` → 重跑 `consistency.py`，报告即自动剔除该人。

---

## 5. 降级回退 A1 静态包（服务机不可用时）

口径完全一致，A1 基线即降级路径：

```powershell
.\.venv\Scripts\python.exe calibration\tools\make_study_package.py `
    --images local-data\screenshots --duration 3 --seed 20260913 --force
# 分发 local-data\calibration-package\ 文件夹；同事本地打开 index.html，下载 JSON 交回；
# 回收后与 web 导出的 JSON 放同一目录跑 consistency.py（数据口径一致）。
```

---

## 6. 安全注记（内网-only）

- **bind**：默认 `127.0.0.1`；内网供多人访问时显式 `--host <内网IP>`，仅内网放行该端口，**严禁公网暴露**。
- **访问控制**：参与者 `session_id`（不可猜 token）；管理端点 `X-Admin-Token`（启动生成，不落盘不入日志）。
- **零外链**：serve 的 `index.html`/`admin.html` 零外链、零 data:URI 大图、零外部 fetch（仅同源 `/api` `/images`）。
- **访问日志脱敏**：只记 `{time, method, path, status}`，不落 IP 原文、不落 session/admin token（路径 token 段遮蔽为 `[redacted]`）。
- **数据**：截图仅 `local-data/web-images/`，标注 `local-data/web-calibration/responses.jsonl`，均 Git 忽略、不入仓库。
- **删除权**：`DELETE /admin/participants/{id}` 删行（重写 JSONL 原子替换）+ 重导出 + 重跑一致性。

---

## 7. 测试

```powershell
# 仅 web 线测试（含 TestClient E2E / 并发 / 删除权 / retire / 零外链 / 种子快照）
.\.venv\Scripts\python.exe -m pytest calibration\tests\test_web_*.py -q

# 全套件（546+ 主套件 + calibration 线）
.\.venv\Scripts\python.exe -m pytest tests calibration\tests -q

# lint
.\.venv\Scripts\python.exe -m ruff check calibration\web
```
