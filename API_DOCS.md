# STP New REST API

六视图更新：新增 `POST /api/render/match` 和检索参数 `include_view_alignment`。接口字段、前端展示示例与旧渲染缓存升级步骤见 [六视图坐标统一与图像配对](VIEW_ALIGNMENT.md)。

> 版本：2.2.0
> 基础地址：`http://服务器IP:8001`
> Swagger：`http://服务器IP:8001/docs`

## 1. 本版数据模型

系统从单一全局零件目录升级为多零件库：

```text
零件库 library_id
├─ 零件记录：文件信息 + STP解析信息 + 索引状态
└─ 独立索引目录
   ├─ 文本 Embedding / Chroma
   ├─ 64维几何 FAISS
   └─ 512维视觉 FAISS
```

每个库的文件、索引和检索范围彼此独立。系统自动提供兼容旧调用的
`default`（默认零件库）。

失败响应统一使用 HTTP 4xx/5xx：

```json
{"success": false, "detail": "错误原因"}
```

## 2. 零件库接口

### 2.1 创建零件库

```http
POST /api/libraries
Content-Type: application/json
```

```json
{
  "name": "示例零件库",
  "description": "通过接口动态创建的零件库"
}
```

响应中的 `library_id` 是上传、建索引和检索时使用的库标识。

### 2.2 查询零件库列表

```http
GET /api/libraries
```

返回每个库的：

- `part_count`：零件数量；
- `indexed_part_count`：已纳入当前索引的零件数量；
- `index.status`：`not_built/building/ready/failed`；
- `text_count/geometric_count/visual_count`：三路索引数量；
- `indexed_at` 和最近错误。

### 2.3 查询单个零件库

```http
GET /api/libraries/{library_id}
```

## 3. 上传、渲染并直接建立三路索引

推荐上传页面只调用一次批量接口。后端执行顺序为：保存文件 → 解析元数据 →
生成六视图 → 一次性重建文本/几何/视觉三路索引 → 返回完整结果。

```http
POST /api/files/upload-batch
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `files` | File[] | 重复提交的 STP/STEP 文件，最多 500 个 |
| `category` | string | 固定为 `library` |
| `library_id` | string | 第 2 节创建的零件库 ID |
| `auto_render` | boolean | 是否在建索引前生成六视图，批量接口默认 `true` |
| `render_force` | boolean | 是否覆盖已有渲染缓存，默认 `false` |
| `include_rotation` | boolean | 是否额外生成旋转 GIF；视觉索引不需要 GIF |
| `rotation_frames` | integer | 旋转 GIF 帧数，范围 12～120，默认 36 |
| `auto_index` | boolean | `true` 时在渲染完成后立即构建三路索引 |

```bash
curl -X POST http://127.0.0.1:8001/api/files/upload-batch \
  -F "category=library" \
  -F "library_id=零件库ID" \
  -F "auto_render=true" \
  -F "render_force=false" \
  -F "include_rotation=false" \
  -F "auto_index=true" \
  -F "files=@part-001.stp" \
  -F "files=@part-002.step"
```

响应：

```json
{
  "success": true,
  "library_id": "abc123",
  "files": [
    {
      "file_id": "零件ID",
      "original_name": "part-001.stp",
      "library_id": "abc123",
      "library_name": "示例零件库",
      "index_status": "ready",
      "indexed_at": "2026-08-06T00:00:00+00:00",
      "metadata": {
        "bbox_dims": [12.0, 35.0, 80.0],
        "num_faces": 24,
        "num_edges": 52,
        "mfg_features": {}
      }
    }
  ],
  "renders": [
    {
      "file_id": "零件ID",
      "status": "ready",
      "views": {
        "front": "/media/renders/libraries/abc123/零件/front.png",
        "back": "/media/renders/libraries/abc123/零件/back.png",
        "top": "/media/renders/libraries/abc123/零件/top.png",
        "bottom": "/media/renders/libraries/abc123/零件/bottom.png",
        "left": "/media/renders/libraries/abc123/零件/left.png",
        "right": "/media/renders/libraries/abc123/零件/right.png"
      },
      "rotation_url": null,
      "cached": false,
      "error": null
    }
  ],
  "index": {
    "status": "ready",
    "text_count": 2,
    "geometric_count": 2,
    "visual_count": 2,
    "indexed_at": "2026-08-06T00:00:00+00:00",
    "error": null
  },
  "index_error": null,
  "processing": {
    "uploaded_count": 2,
    "metadata_parsed_count": 2,
    "rendered_count": 2,
    "render_failed_count": 0,
    "text_index_count": 2,
    "geometric_index_count": 2,
    "visual_index_count": 2,
    "elapsed_seconds": 18.6,
    "complete": true
  }
}
```

图片字段含义：

| 字段 | 含义 |
|---|---|
| `front` | 主视图/前视图 |
| `back` | 后视图 |
| `top` | 俯视图/顶视图 |
| `bottom` | 仰视图/底视图 |
| `left` | 左视图 |
| `right` | 右视图 |
| `rotation_url` | 可选旋转 GIF；未生成时为 `null` |
| `cached` | 是否复用已有六视图缓存 |
| `error` | 单个零件渲染失败原因 |

图片 URL 是相对后端地址。浏览器应拼接 API 地址，例如：
`http://服务器IP:8001` + `views.front`。

渲染和 `auto_index` 均为同步操作，零件很多时请求会持续较长时间。若单个零件
渲染失败，文件仍会保存并继续构建文本/几何索引；此时 `processing.complete=false`，
且视觉索引数量可能少于文件数。若整个建库失败，通过 `index.status=failed` 和
`index_error` 给出原因。

启用 `auto_render=true` 与 `auto_index=true` 时，只有所有零件渲染成功、索引状态为
`ready`，且视觉索引数量不少于文本索引数量，`processing.complete` 才会为 `true`。

单文件接口 `POST /api/files/upload` 同样支持 `library_id` 和
`auto_render`、`include_rotation` 和 `auto_index`，但连续上传多个文件时不要每个文件
都重建索引。

查询件上传不属于零件库：

```bash
curl -X POST http://127.0.0.1:8001/api/files/upload \
  -F "category=query" -F "file=@query.stp"
```

## 4. 零件管理接口

### 4.1 分页查询全部零件信息

```http
GET /api/parts?library_id={library_id}&keyword=零件&offset=0&limit=50
```

响应包含文件信息、所属零件库、STP 解析元数据、索引状态和时间，可直接用于
“零件管理”页面。

```json
{
  "success": true,
  "library_id": "abc123",
  "total": 120,
  "offset": 0,
  "limit": 50,
  "items": []
}
```

不传 `library_id` 时返回所有零件库的零件。

### 4.2 零件详情

```http
GET /api/parts/{file_id}
```

### 4.3 原文件及解析详情

```http
GET /api/files/{file_id}/download
GET /api/files/{file_id}/inspect
```

### 4.4 删除零件

```http
DELETE /api/parts/{file_id}
```

该接口只接受零件库文件的 `file_id`，并同步删除：

- 零件管理中的目录记录；
- 服务器上的 STP/STEP 原文件；
- 六视图与旋转 GIF 渲染缓存；
- 文本 Embedding、几何 FAISS、视觉 FAISS 三路索引记录。

前端调用示例：

```javascript
await fetch(`http://服务器IP:8001/api/parts/${fileId}`, {
  method: "DELETE"
});
```

成功响应示例：

```json
{
  "success": true,
  "file_id": "零件ID",
  "original_name": "part.stp",
  "library_id": "abc123",
  "file_deleted": true,
  "render_deleted": true,
  "index_removed": {
    "text": true,
    "geometric": true,
    "visual": true
  },
  "index": {
    "status": "ready",
    "text_count": 19,
    "geometric_count": 19,
    "visual_count": 19,
    "indexed_at": "2026-08-13T00:00:00+00:00",
    "error": null
  },
  "remaining_parts": 19,
  "message": "零件 part.stp 已删除，零件库剩余 19 个零件"
}
```

原有文件列表仍保留，并增加零件库筛选：

```http
GET /api/files?category=library&library_id={library_id}
```

### 4.5 `metadata` 字段说明

Swagger 2.2.0 已为返回模型中的字段加入中文 `description`。截图中常见字段含义如下：

| 字段 | 含义 |
|---|---|
| `entity_counts` | 各类 STEP 实体及其数量 |
| `total_entities` | STEP 实体总数 |
| `num_faces/num_edges/num_vertices` | 面、边、顶点数量 |
| `face_types` | 平面、圆柱面、样条面等面类型数量 |
| `edge_types` | 直线、圆、样条线等边类型数量 |
| `bbox_dims` | 包围盒三边尺寸，已从小到大排序，单位通常为 mm |
| `bbox_diagonal` | 包围盒空间对角线长度 |
| `radii` | 圆曲线半径列表 |
| `cylinder_radii` | 圆柱面半径列表 |
| `aspect_ratios` | 基于包围盒计算的无量纲尺寸比例 |
| `total_surface_area_estimate` | 估算表面积，不是精确 CAD 面积 |
| `volume_estimate` | 估算体积，不是精确 CAD 体积 |
| `compactness` | 估算体积与表面积形成的紧凑度指标 |
| `rotational_symmetry_score` | 旋转对称性估计得分 |
| `mfg_features` | 孔、槽、型腔、凸台、加强筋、倒角、圆角、螺纹等制造特征 |
| `mfg_feature_vector` | 制造特征相似度计算使用的数值向量，前端通常不直接展示 |

这些值来自 STEP 文本和拓扑关系的解析及启发式估算。精确体积、面积或加工特征仍应
以专业 CAD 内核计算结果为准。

## 5. 单独建立或重建索引

如果上传时 `auto_index=false`，调用：

```http
POST /api/build-index
Content-Type: application/json
```

```json
{"library_id": "abc123"}
```

也可以使用语义更明确的路径：

```http
POST /api/libraries/{library_id}/build-index

{}
```

响应分别报告三路索引数量：

```json
{
  "success": true,
  "library_id": "abc123",
  "library_name": "示例零件库",
  "directory": "/home/user/stpnew/runtime/library/abc123",
  "file_count": 120,
  "index_count": 120,
  "text_index_count": 120,
  "geometric_index_count": 120,
  "visual_index_count": 118,
  "elapsed_seconds": 63.42,
  "message": "三路索引构建完成：文本 120，几何 120，视觉 118"
}
```

视觉数量可能少于文本数量：没有可用六视图的零件无法建立视觉向量。推荐直接使用
第 3 节的一体化上传接口并传 `auto_render=true`、`auto_index=true`。Embedding 索引
需要正确配置 `API_KEY/BASE_URL/EMBEDDING_MODEL`。

## 6. 按零件库进行相似检索

```http
POST /api/search
Content-Type: application/json
```

前端页面字段对应：

- “选择多个零件库” → `library_ids`
- “返回数量” → `result_limit`，范围 1～50；未传时默认 10

```json
{
  "file_id": "查询件上传后返回的ID",
  "library_ids": ["u_library_id", "l_library_id", "p_library_id"],
  "result_limit": 10,
  "coarse_top": 30,
  "use_three_way": true,
  "use_llm_rerank": true,
  "use_vision": false,
  "use_geometric": true,
  "use_feature_extraction": false,
  "use_3d_vision": false,
  "auto_render": false,
  "fusion_top_k": 20
}
```

只有 `index.status=ready` 的零件库允许检索。只要任意一个所选库尚未完成索引，
接口就返回 409 并列出未就绪的库。若查询零件本身来自选定零件库，
后端会自动从结果中排除查询件。

当 LLM 精排返回的排名条数少于 `result_limit` 时，后端会按召回综合分数
补齐未被模型列出的候选。因此，只要零件库中有足够的非查询候选，接口会
稳定返回请求数量；零件库可用候选不足时返回实际可用数量。

`library_ids` 最多选择20个独立零件库。后端分别在各库的隔离索引中召回，
合并候选、排除查询件和重复路径后，再按照 `composite_score` 统一降序排序。
旧字段 `library_id` 继续支持；未传 `library_ids` 时按单库模式运行。

响应：

```json
{
  "success": true,
  "library_id": null,
  "library_name": null,
  "library_ids": ["u_library_id", "l_library_id"],
  "library_names": ["U型零件库", "L型零件库"],
  "query_file_id": "query-file-id",
  "query_filename": "query.stp",
  "elapsed_seconds": 2.35,
  "total_results": 10,
  "result_limit": 10,
  "results": [
    {
      "rank": 1,
      "file_id": "candidate-file-id",
      "library_id": "abc123",
      "filename": "part-001.stp",
      "composite_score": 0.93,
      "hybrid_similarity": 0.93
    }
  ],
  "report_url": null
}
```

旧字段 `final_top` 仍然支持；同时传入时优先使用 `result_limit`。

`results` 已严格按照 `composite_score` 从高到低排列。该字段是最终综合分数，
与兼容字段 `hybrid_similarity` 数值相同；前端结果卡片最上方应显示
`(item.composite_score * 100).toFixed(2) + '%'`，不要使用
`embedding_similarity` 作为顶部总分。

## 7. 在线渲染

```http
POST /api/render
Content-Type: application/json
```

```json
{
  "file_id": "零件ID",
  "force": false,
  "include_rotation": true,
  "rotation_frames": 36
}
```

返回六视图 URL 和可选旋转 GIF。服务器需要 PyVista、pythonocc-core 和离屏渲染环境。

## 8. 健康检查

```http
GET /api/status
```

除能力状态外，新增：

- `library_count`
- `total_parts`
- `index_count`：所有零件库文本索引数量之和

## 9. 前端推荐调用顺序

1. 页面加载时调用 `GET /api/libraries`，填充“选择零件库”下拉框。
2. 新零件库调用 `POST /api/libraries`。
3. 上传页调用 `POST /api/files/upload-batch`，携带 `library_id`、
   `auto_render=true` 和 `auto_index=true`。等待响应中的 `processing.complete=true`。
4. 零件管理页调用 `GET /api/parts`。
5. 查询件以 `category=query` 上传，得到 `file_id`。
6. 相似检索携带查询 `file_id`、选中的 `library_id` 和滑块
   `result_limit`。

可直接复用 `frontend/api.ts` 中的：

- `createLibrary`
- `listLibraries`
- `uploadLibraryParts`
- `listParts`
- `buildIndex`
- `searchSimilar`

## 10. LTR 接口

LTR 路径保持不变：

- `POST /api/ltr/train`
- `POST /api/ltr/evaluate`
- `GET /api/ltr/feature-importance`
- `POST /api/ltr/predict`

未安装 LTR 依赖时不影响零件库、上传、三路索引、渲染和普通相似检索。
