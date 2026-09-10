# 六视图坐标统一与图像配对

本次修改保持 API 端口 8001，原有上传、检索、渲染接口仍可调用。新增 `POST /api/render/match`，以及检索可选字段 `include_view_alignment`。渲染图片仍使用 front/back/top/bottom/left/right 六个名称。

## 问题与方案

原实现保留 STEP 的世界坐标和初始姿态；通过相机 azimuth/elevation 切换视图，未显式固定正交投影和各视图缩放。不同来源 CAD 的同名视图因此不一定是同一方向。另一个问题是渲染返回顺序与查图顺序不同，LLM 标签却按数组位置生成，导致方向标签错位。

处理流程：

1. STEP 经 OCC 转为三角网格。仅变换渲染网格副本，不改变原始 CAD 文件、解析出的尺寸或几何索引。
2. 用三角形面积加权聚合无向面法线，优先选择主要平面方向及近似垂直方向。曲面缺乏主要平面时使用面积加权表面 PCA。面积权重可减少三角网格密度对坐标轴的影响。
3. 按候选坐标下的包围盒尺寸排序：最长方向为 Y，中间为 X，最短为 Z；保持右手坐标系。利用非对称三阶矩处理部分正负号歧义，再将包围盒中心平移到原点。
4. 六台固定相机使用正交投影、固定 view-up 和同一 parallel_scale。每个方向使用独立的离屏渲染窗口，并显式设置相机位置、焦点和向上方向，避免部分 Linux Mesa/EGL 环境复用第一帧而生成六张相同图片。每件零件最长尺寸约占画面 83%，六张图之间保持同一比例；不同零件按自身尺寸归一化，实际大小应读取尺寸字段，不能按像素推断。
5. 对查询件和候选件构建 6×6×4 匹配分数：每个视图组合考虑 0/90/180/270° 图片旋转。85% 前景轮廓 IoU（保留孔洞），15% 灰度一致性。匹配特征会平移居中，保留统一尺度。
6. 默认枚举 24 种行列式为 +1 的三维轴旋转，同时决定六视图对应关系和各图旋转角度。拒绝镜像，前后、上下、左右必须一致。
7. 同时计算匈牙利类线性分配结果作为独立匹配的参考上界。SciPy 的 `linear_sum_assignment` 实际采用改进 Jonker–Volgenant 算法，解决相同的最优一一分配问题。
8. 输出按查询件方向命名的候选图片副本。多模态检索在完整六视图可用时，也会先配对再发送候选图片；视图标签根据文件名生成。方向改变后的图片不会复用旧的 LLM 特征缓存。

相机约定（位置方向是从物体中心指向相机）：

| 视图 | 相机位置方向 | 图片向上方向 |
| --- | --- | --- |
| front | +Z | +Y |
| back | −Z | +Y |
| top | +Y | −Z |
| bottom | −Y | +Z |
| left | −X | +Y |
| right | +X | +Y |

## 前端详情页调用

建议在打开零件对比详情时调用，以避免一次搜索为所有结果补渲染。

```http
POST /api/render/match
Content-Type: application/json
```

```json
{
  "query_file_id": "查询零件ID",
  "candidate_file_id": "候选零件ID",
  "method": "rigid24",
  "force": false
}
```

`force=false` 会检查渲染版本、源文件路径/大小/修改时间。当前修复后的渲染版本为 `canonical-ortho-v2`；旧版、没有 manifest 或 `canonical-ortho-v1` 的缓存会自动重新渲染。只有渲染和本地图像配对，不调用模型 API，不要求零件库已建立索引。

| 返回字段 | 含义 / 前端用法 |
| --- | --- |
| `query_views` | 左侧查询件的六张标准图片 URL |
| `candidate_views` | 候选件配对之前的标准图片 URL |
| `aligned_views` | 右侧候选件已旋转的六张图片 URL，以查询件方向命名；直接与 query_views 同名展示 |
| `pairs` | 每对视图的源方向、逆时针旋转角度和匹配分数 |
| `score` | 六对图像平均匹配分数，0~1；不是检索综合分数、准确率或实际几何等价概率 |
| `hungarian_score` | 不考虑全局三维一致性的参考分数上界 |
| `rotation_matrix` | 将候选标准坐标转换到查询标准坐标的 3×3 矩阵（列向量约定），不含平移和缩放 |
| `rigid_consistent` | rigid24 模式为 true；表示施加了全局三维旋转约束 |
| `score_gap` | 最优与次优刚性方向分差；小于0.015时标记方向有歧义 |
| `ambiguous` | 对称件可能有多个等价方向；true 不代表请求失败 |
| `warning` | 非刚性配对或方向歧义解释 |

例如 `pairs` 的一项为 `{"query_view":"front","candidate_view":"right","rotation_degrees_ccw":90,"score":0.94}`，表示候选右视图逆时针转90°后对应查询前视图。`aligned_views.front` 已完成旋转，前端不能再转一次。

```javascript
const response = await fetch(`${API_BASE}/api/render/match`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query_file_id: queryId, candidate_file_id: candidateId })
});
const data = await response.json();
if (!response.ok) throw new Error(data.detail || "六视图配对失败");
const order = ["front", "right", "back", "left", "top", "bottom"];
const cards = order.map(view => ({
  view,
  queryImage: `${API_BASE}${data.query_views[view]}`,
  candidateImage: `${API_BASE}${data.aligned_views[view]}`
}));
```

如果需要直接在检索返回中携带配对信息，在原 `POST /api/search` 请求上增加：

```json
{
  "file_id": "查询零件ID",
  "library_ids": ["零件库ID"],
  "result_limit": 10,
  "include_view_alignment": true
}
```

每个结果包含 `view_alignment`；失败时为 null，并提供 `view_alignment_error`，原检索结果仍返回。此字段默认 false，以保留原来的检索延迟。返回顺序仍按原 composite_score，配对分数不修改排序。

`method="hungarian"` 可做试验比较。它允许任意一一对应及逐图旋转，可能将物理上不一致的六张图拼在一起，因此不作为默认方向统一方案。

## 服务器更新与历史数据

需要上传这些运行文件（不是只上传 app）：

```text
stpnew/view_alignment.py          新增：坐标归一化、正交渲染、图像配对
stpnew/stp_similarity.py          统一渲染入口、缓存版本、LLM视图配对与标签修复
stpnew/app/schemas.py             配对请求/响应与检索字段
stpnew/app/services_impl.py       配对渲染服务、URL转换
stpnew/app/routes_impl.py         新接口与检索配对返回
stpnew/render_cli.py              按零件库升级渲染缓存
```

新代码使用现有依赖 numpy/scipy/pillow/pyvista/pythonocc-core，不需要新增大模型服务。可同时上传 `tests/test_view_alignment.py`、`tests/verify_render_alignment.py`、`smoke_test.py` 和本文档用于验证。

服务器语法检查：

```bash
cd /home/shensuan/stpnew
/home/shensuan/.conda/envs/name/bin/python -m py_compile view_alignment.py stp_similarity.py app/schemas.py app/services_impl.py app/routes_impl.py render_cli.py
```

随后按原部署方式重启8001服务。原 `.env`、上传文件、零件库记录和索引目录不需要被本地版本覆盖。

建议在维护窗口内，逐库升级已有图片（将示例ID替换为实际 library_id；执行期间暂停该库检索/上传）：

```bash
python render_cli.py --library-id YOUR_LIBRARY_ID --force
```

仅刷新图片不会刷新已有视觉向量。渲染成功后，按原接口重新建库：

```bash
curl -sS -X POST http://127.0.0.1:8001/api/build-index \
  -H 'Content-Type: application/json' \
  -d '{"library_id":"YOUR_LIBRARY_ID"}'
```

注意：项目现有 build-index 会重建三路索引并调用配置的 Embedding API。升级前应备份该库索引；若模型接口不可用，先恢复接口再重建，避免原索引被清空后建库失败。这里只提供操作步骤，不会自动在服务器触发重建。

历史查询件可以通过原 `/api/render` 或新 `/api/render/match` 自动升级。已有 PNG 本身无法还原原来的透视畸变或任意三维倾斜，必须从 STEP 重新渲染。视图配对生成的缓存按图像路径、修改时间、大小与算法版本区分，存放在 renders/alignments，原图片不会被配对覆盖。

## 验证与边界

- 单元测试覆盖24种合法轴旋转、图片旋转方向、独立配对与刚性配对区别、空白/缺失视图、缓存失效、离屏相机实际切换和LLM方向标签。
- API 冒烟测试覆盖新路由、无效方法/文件ID、搜索携带配对以及渲染失败后的搜索降级。
- 实际渲染验证使用本地测试件3的OCC网格，以及同一网格旋转(33°, −47°, 71°)并平移后的版本，检查六张图片对应、包围盒尺度和输入网格不被修改。
- 本机 CAD 环境的合并渲染进程曾提前退出；实际零件验证采用 CAD 环境导出STL、基础环境离屏渲染的分阶段方式完成。服务器完整环境仍需执行下方验证或新接口验收；没有在远程服务器替用户部署。

```bash
python -m unittest tests.test_view_alignment tests.test_rerank_completion tests.test_llm_gateway
python smoke_test.py
python tests/verify_render_alignment.py --step /path/to/valid_part.stp
```

该方法可消除平移、主要方向及轴正负歧义，但不能保证所有不同设计的零件都拥有唯一的“工程主视图”。对完全对称件、圆形件、主平面差异较大的零件，应结合 ambiguous、score_gap 和配对图人工检查。刚性24配对只处理轴置换与90°倍数图像旋转；主方向归一化残留的任意角度差异不会被它凭空修正。

主要依据：[PyVista相机投影](https://docs.pyvista.org/api/core/_autosummary/pyvista.camera.parallel_projection)、[固定正交缩放](https://docs.pyvista.org/api/core/_autosummary/pyvista.camera.parallel_scale)、[SciPy线性最优分配](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)。PCA方向会受形状影响，见[Open3D定向包围盒说明](https://www.open3d.org/docs/release/python_api/open3d.geometry.OrientedBoundingBox.html)；本实现优先使用面积加权主要面法线，无需安装Open3D。
