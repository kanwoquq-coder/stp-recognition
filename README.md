# STP New

基于当前 `stp_similarity.py` 重新封装的独立 FastAPI 服务，默认端口为 **8001**。支持多零件库管理、上传后自动解析与六视图渲染、文本/几何/视觉三路索引、零件元数据持久化、按库相似检索和可选 LTR。

## 目录

```text
stpnew/
├─ app/                    FastAPI 配置、模型、路由和服务层
├─ frontend/api.ts         浏览器/前端请求封装
├─ runtime/                上传、零件库、索引、渲染和模型运行数据
├─ tests/smoke_test.py     不调用外部模型的接口冒烟测试
├─ stp_similarity.py       当前检索核心的独立副本
├─ stp_ltr.py              LTR 核心
├─ recall_debugger.py      召回诊断
├─ render_cli.py           从 step.py 整理的批量渲染入口
├─ run.py                  服务入口
├─ requirements.txt        Python 依赖
└─ API_DOCS.md             接口说明
```

## 安装

推荐使用 Python 3.10。在线 CAD 渲染依赖 `pythonocc-core`，建议使用 Conda：

```bash
conda env create -f environment.yml
conda activate stpnew
```

若暂时不需要在线渲染，也可直接：

```bash
pip install -r requirements.txt
```

`pythonocc-core` 通常需要额外执行：

```bash
conda install -c conda-forge pythonocc-core
```

## 配置和启动

复制 `.env.example` 为 `.env`，至少填写 Embedding 服务的 `API_KEY`。精排可使用独立的 OpenAI 兼容 Qwen 服务；例如：

```env
LLM_API_KEY=EMPTY
LLM_BASE_URL=http://10.100.0.34:8223/v1
MODEL=Qwen3.5-27B
LLM_THINKING=off
LLM_RESPONSE_FORMAT=auto
LLM_TIMEOUT_SECONDS=180
LLM_MAX_RETRIES=1
```

模型网关会在服务不支持 `response_format` 时自动降级；图片请求失败但文本请求正常时，会停止重复发送图片并回退到文本精排。这些设置只影响后端模型调用，现有 HTTP 接口的入参和出参不变。

```bash
python run.py
```

启动后：

- 服务：`http://localhost:8001`
- Swagger：`http://localhost:8001/docs`
- 健康检查：`http://localhost:8001/api/status`

## 推荐前后端流程

1. 调用 `POST /api/libraries` 创建零件库，保存返回的 `library_id`。
2. 调用 `POST /api/files/upload-batch`，表单中传 `category=library`、`library_id`、`auto_render=true` 和 `auto_index=true`；后端依次保存、解析、生成六视图，再一次性构建文本/几何/视觉索引，并返回 `renders/index/processing`。
3. 零件管理页调用 `GET /api/parts?library_id=...`，读取文件信息、STP 解析元数据和索引状态。
4. 使用 `POST /api/files/upload`，以 `query` 类别上传查询零件并取得 `file_id`。
5. 调用 `POST /api/search`，传查询 `file_id`、目标零件库数组 `library_ids` 和 `result_limit`；旧前端仍可传单个 `library_id`。
6. 前端可直接复用 `frontend/api.ts`。

浏览器所在电脑的本地路径无法被远端服务器直接读取，因此标准接口使用 `file_id`。只有后端与本地零件库在同一台可信机器时，才将 `.env` 中 `ALLOW_LOCAL_PATHS=true`，并通过 `LIBRARY_DIR` 指向本地测试库。

## 使用现有本地零件库

```env
ALLOW_LOCAL_PATHS=true
LIBRARY_DIR=C:/Users/phillip/Desktop/project/小支架零件总成
```

随后调用：

```bash
curl -X POST http://localhost:8001/api/build-index \
  -H "Content-Type: application/json" \
  -d "{}"
```

批量离线渲染：

```bash
python render_cli.py "C:/path/to/stp-library" --output "C:/path/to/pictures"
```

## 部署注意

- 开放服务器 TCP 端口 `8001`。
- Docker 映射使用 `-p 8001:8001`。
- Linux 无显示环境运行 PyVista 时需配置 EGL/OSMesa 或 `xvfb-run`。
- 不要把真实 API Key 写入 `.env.example` 或提交到版本库。
- LTR 标注 JSON 中引用的零件路径必须能被后端访问；远端部署时应改为服务器路径。
