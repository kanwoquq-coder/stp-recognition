from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FileCategory = Literal["query", "library", "reference", "annotation"]
IndexStatus = Literal["not_built", "building", "ready", "failed"]
RenderStatus = Literal["ready", "skipped", "failed"]


class ManufacturingFeatures(BaseModel):
    """由 STEP 拓扑关系启发式识别的可制造特征。"""

    model_config = ConfigDict(extra="allow")

    through_holes: int = Field(default=0, description="贯穿孔数量（启发式估算）")
    blind_holes: int = Field(default=0, description="盲孔数量（启发式估算）")
    avg_hole_depth_ratio: float = Field(default=0.0, description="孔深与零件尺度的平均比值")
    hole_aspect_ratio: float = Field(default=0.0, description="孔深径比的归一化估计值")
    slots: int = Field(default=0, description="槽特征数量")
    pockets: int = Field(default=0, description="型腔或口袋特征数量")
    avg_slot_depth_ratio: float = Field(default=0.0, description="槽深与零件尺度的平均比值")
    avg_slot_lw_ratio: float = Field(default=0.0, description="槽长宽比的平均值")
    bosses: int = Field(default=0, description="凸台数量")
    avg_boss_height_ratio: float = Field(default=0.0, description="凸台高度与零件尺度的平均比值")
    boss_density: float = Field(default=0.0, description="凸台数量相对于拓扑复杂度的密度")
    ribs: int = Field(default=0, description="加强筋数量")
    rib_thickness_ratio: float = Field(default=0.0, description="加强筋厚度比例估计值")
    chamfers: int = Field(default=0, description="倒角数量")
    fillets: int = Field(default=0, description="圆角数量")
    avg_chamfer_angle: float = Field(default=0.0, description="平均倒角角度估计值")
    avg_fillet_radius_ratio: float = Field(default=0.0, description="圆角半径与零件尺度的平均比值")
    threads: int = Field(default=0, description="螺纹特征数量")
    has_thread: int = Field(default=0, description="是否检测到螺纹，0=否，1=是")
    unique_feature_types: int = Field(default=0, description="检测到的制造特征类型数量")
    feature_density: float = Field(default=0.0, description="制造特征密度")
    feature_diversity: float = Field(default=0.0, description="制造特征多样性得分")
    complexity_score: float = Field(default=0.0, description="制造复杂度综合得分")
    manufacturing_symmetry: float = Field(default=0.0, description="制造特征对称性得分")
    mfg_feature_vector: list[float] = Field(
        default_factory=list, description="用于制造特征相似度计算的数值向量"
    )


class PartMetadata(BaseModel):
    """上传后解析并保存在零件管理中的 STEP 几何与制造信息。"""

    model_config = ConfigDict(extra="allow")

    filename: str = Field(default="", description="上传后的服务器文件名")
    filepath: str = Field(default="", description="服务器内部文件路径，前端通常无需使用")
    entity_counts: dict[str, int] = Field(default_factory=dict, description="各 STEP 实体类型及数量")
    total_entities: int = Field(default=0, description="STEP 实体总数")
    num_faces: int = Field(default=0, description="面数量")
    num_edges: int = Field(default=0, description="边数量")
    num_vertices: int = Field(default=0, description="顶点数量")
    num_edge_loops: int = Field(default=0, description="边环数量")
    num_oriented_edges: int = Field(default=0, description="有向边数量")
    num_closed_shells: int = Field(default=0, description="闭合壳数量")
    num_solids: int = Field(default=0, description="实体数量")
    euler: int = Field(default=0, description="欧拉示性数：顶点数-边数+面数")
    face_types: dict[str, int] = Field(default_factory=dict, description="平面、圆柱面、样条面等面类型数量")
    edge_types: dict[str, int] = Field(default_factory=dict, description="直线、圆、样条线等边类型数量")
    num_points: int = Field(default=0, description="STEP 中解析到的笛卡尔点数量")
    num_bbox_points: int = Field(default=0, description="参与包围盒计算的有效点数量")
    bbox_source: str = Field(default="", description="包围盒点来源，例如 vertex 或 cartesian_fallback")
    bbox_dims: list[float] = Field(default_factory=list, description="包围盒三边尺寸，已按从小到大排序，单位通常为 mm")
    bbox_diagonal: float = Field(default=0.0, description="包围盒空间对角线长度")
    point_distribution_ratios: list[float] = Field(default_factory=list, description="点云主方向特征值比例，用于描述形状分布")
    radii: list[float] = Field(default_factory=list, description="解析到的圆曲线半径列表")
    unique_radii: list[float] = Field(default_factory=list, description="去重后的圆曲线半径列表")
    num_unique_radii: int = Field(default=0, description="不同圆半径的数量")
    cylinder_radii: list[float] = Field(default_factory=list, description="圆柱面半径列表")
    aspect_ratios: list[float] = Field(default_factory=list, description="基于包围盒计算的无量纲长宽比例")
    face_area_cv_proxy: float = Field(default=0.0, description="面尺寸离散程度的近似指标")
    face_type_entropy: float = Field(default=0.0, description="面类型分布熵，越高表示类型越丰富")
    total_surface_area_estimate: float = Field(default=0.0, description="基于包围盒和面数估算的表面积，不是精确 CAD 面积")
    volume_estimate: float = Field(default=0.0, description="基于包围盒估算的体积，不是精确 CAD 体积")
    compactness: float = Field(default=0.0, description="估算体积与估算表面积的紧凑度指标")
    edge_face_ratio: float = Field(default=0.0, description="边数与面数之比")
    vertex_face_ratio: float = Field(default=0.0, description="顶点数与面数之比")
    curve_complexity: float = Field(default=0.0, description="样条线和椭圆边占比形成的曲线复杂度")
    feature_richness: float = Field(default=0.0, description="不同圆半径和圆柱半径形成的特征丰富度")
    rotational_symmetry_score: float = Field(default=0.0, description="基于圆柱面比例和包围盒估算的旋转对称性")
    mfg_features: ManufacturingFeatures = Field(
        default_factory=ManufacturingFeatures, description="启发式识别的制造特征明细"
    )
    mfg_feature_vector: list[float] = Field(
        default_factory=list, description="制造特征向量，与 mfg_features 中的向量一致"
    )
    parse_error: str | None = Field(default=None, description="解析失败时的错误信息；正常情况下为 null")


class ViewImages(BaseModel):
    """零件六视图的浏览器可访问 URL。"""

    front: str | None = Field(default=None, description="主视图/前视图图片 URL")
    back: str | None = Field(default=None, description="后视图图片 URL")
    top: str | None = Field(default=None, description="俯视图/顶视图图片 URL")
    bottom: str | None = Field(default=None, description="仰视图/底视图图片 URL")
    left: str | None = Field(default=None, description="左视图图片 URL")
    right: str | None = Field(default=None, description="右视图图片 URL")


class PartRenderResult(BaseModel):
    file_id: str = Field(description="对应上传零件的文件 ID")
    status: RenderStatus = Field(description="渲染状态：ready、skipped 或 failed")
    views: ViewImages = Field(default_factory=ViewImages, description="六视图图片地址")
    rotation_url: str | None = Field(default=None, description="可选的旋转 GIF 地址；未生成时为 null")
    cached: bool = Field(default=False, description="是否直接复用了已有渲染缓存")
    error: str | None = Field(default=None, description="渲染失败原因；成功时为 null")


class IngestionSummary(BaseModel):
    uploaded_count: int = Field(default=0, description="本次成功保存的文件数量")
    metadata_parsed_count: int = Field(default=0, description="成功解析 STEP 元数据的文件数量")
    rendered_count: int = Field(default=0, description="成功获得六视图的文件数量")
    render_failed_count: int = Field(default=0, description="六视图渲染失败的文件数量")
    text_index_count: int = Field(default=0, description="建库完成后的文本索引数量")
    geometric_index_count: int = Field(default=0, description="建库完成后的几何索引数量")
    visual_index_count: int = Field(default=0, description="建库完成后的视觉索引数量")
    elapsed_seconds: float = Field(default=0.0, description="上传、解析、渲染和建库总耗时，单位秒")
    complete: bool = Field(default=False, description="渲染和索引是否全部完成；部分失败时为 false")


class StatusResponse(BaseModel):
    status: str = Field(default="ok", description="服务状态，正常时为 ok")
    version: str = Field(description="后端接口版本")
    port: int = Field(description="后端监听端口")
    engine_initialized: bool = Field(description="是否已初始化至少一个检索引擎")
    index_count: int = Field(description="所有零件库的文本索引总数量")
    has_api_key: bool = Field(description="服务器是否已配置模型 API Key")
    has_faiss: bool = Field(description="是否具备几何/视觉 FAISS 索引能力")
    has_render: bool = Field(description="是否具备 STEP 六视图渲染能力")
    has_ltr: bool = Field(description="是否安装 LTR 模块依赖")
    ltr_loaded: bool = Field(description="是否已加载训练完成的 LTR 模型")
    allow_local_paths: bool = Field(description="是否允许请求直接传服务器本地路径")
    library_count: int = Field(default=0, description="零件库数量")
    total_parts: int = Field(default=0, description="零件管理中的零件总数")


class IndexSummary(BaseModel):
    status: IndexStatus = Field(default="not_built", description="索引状态：未建、构建中、可用或失败")
    text_count: int = Field(default=0, description="文本 Embedding 索引数量")
    geometric_count: int = Field(default=0, description="64 维几何 FAISS 索引数量")
    visual_count: int = Field(default=0, description="512 维视觉 FAISS 索引数量")
    indexed_at: str | None = Field(default=None, description="最近一次成功建库时间，ISO 8601 格式")
    error: str | None = Field(default=None, description="最近一次建库错误；成功时为 null")


class LibraryCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="零件库名称，供前端下拉框显示")
    description: str = Field(default="", max_length=500, description="零件库用途或内容说明")


class LibraryItem(BaseModel):
    library_id: str = Field(description="零件库唯一 ID，搜索请求的 library_id 使用此值")
    name: str = Field(description="零件库名称")
    description: str = Field(default="", description="零件库说明")
    created_at: str = Field(description="创建时间，ISO 8601 格式")
    updated_at: str = Field(description="最近更新时间，ISO 8601 格式")
    part_count: int = Field(default=0, description="库内零件数量")
    indexed_part_count: int = Field(default=0, description="已纳入当前索引的零件数量")
    index: IndexSummary = Field(default_factory=IndexSummary, description="该零件库的三路索引状态")


class FileItem(BaseModel):
    file_id: str = Field(description="文件唯一 ID；查询零件时作为 file_id 提交")
    original_name: str = Field(description="用户上传时的原始文件名")
    category: FileCategory = Field(description="文件用途：query、library、reference 或 annotation")
    size: int = Field(description="文件大小，单位字节")
    content_type: str = Field(description="上传文件的 MIME 类型")
    created_at: str = Field(description="上传时间，ISO 8601 格式")
    download_url: str = Field(description="原文件下载地址")
    library_id: str | None = Field(default=None, description="所属零件库 ID；查询件等非库文件为 null")
    library_name: str | None = Field(default=None, description="所属零件库名称")
    index_status: IndexStatus = Field(default="not_built", description="该零件当前的索引状态")
    indexed_at: str | None = Field(default=None, description="该零件最近纳入索引的时间")
    metadata: PartMetadata | None = Field(default=None, description="STEP 几何、拓扑和制造特征解析结果")


class UploadResponse(BaseModel):
    success: bool = Field(default=True, description="文件是否保存成功")
    file: FileItem = Field(description="保存后的文件和零件解析信息")
    render: PartRenderResult | None = Field(default=None, description="该零件的六视图渲染结果")
    index: IndexSummary | None = Field(default=None, description="所属零件库的三路索引结果")
    index_error: str | None = Field(default=None, description="建库错误；成功时为 null")
    processing: IngestionSummary | None = Field(default=None, description="上传、渲染和建库处理汇总")


class BatchUploadResponse(BaseModel):
    success: bool = Field(default=True, description="文件是否已完成保存；渲染部分失败时仍可能为 true")
    files: list[FileItem] = Field(description="本次上传保存的所有零件信息")
    library_id: str | None = Field(default=None, description="目标零件库 ID")
    renders: list[PartRenderResult] = Field(default_factory=list, description="每个上传零件对应的六视图渲染结果")
    index: IndexSummary | None = Field(default=None, description="目标零件库的三路索引结果")
    index_error: str | None = Field(default=None, description="建库错误；成功时为 null")
    processing: IngestionSummary = Field(default_factory=IngestionSummary, description="一体化入库流程汇总")


class PartListResponse(BaseModel):
    success: bool = Field(default=True, description="查询是否成功")
    library_id: str | None = Field(default=None, description="当前筛选的零件库 ID；未筛选时为 null")
    total: int = Field(description="符合条件的零件总数")
    offset: int = Field(description="当前分页起始位置")
    limit: int = Field(description="当前分页大小")
    items: list[FileItem] = Field(description="当前页零件信息")


class DeletePartResponse(BaseModel):
    success: bool = Field(default=True, description="删除是否完成")
    file_id: str = Field(description="被删除零件的文件 ID")
    original_name: str = Field(description="被删除零件的原始文件名")
    library_id: str = Field(description="被删除零件所属零件库 ID")
    file_deleted: bool = Field(description="STP/STEP 原文件是否已删除")
    render_deleted: bool = Field(description="六视图和旋转 GIF 缓存是否已删除")
    index_removed: dict[str, bool] = Field(
        default_factory=dict,
        description="三路索引删除结果，键为 text、geometric、visual",
    )
    index: IndexSummary = Field(description="删除完成后零件库的索引状态及数量")
    remaining_parts: int = Field(description="删除后零件库剩余零件数量")
    message: str = Field(description="删除结果摘要")


class InspectResponse(BaseModel):
    success: bool = Field(default=True, description="解析是否成功")
    file_id: str = Field(description="被解析的文件 ID")
    metadata: PartMetadata = Field(description="STEP 几何、拓扑和制造特征")


class FileReference(BaseModel):
    file_id: str | None = Field(default=None, description="已上传文件的 ID；浏览器前端推荐使用")
    file_path: str | None = Field(default=None, description="服务器本地文件路径；仅 ALLOW_LOCAL_PATHS=true 时可用")

    @model_validator(mode="after")
    def validate_reference(self):
        if bool(self.file_id) == bool(self.file_path):
            raise ValueError("file_id 和 file_path 必须且只能提供一个")
        return self


class RenderRequest(FileReference):
    force: bool = Field(default=False, description="是否忽略缓存并强制重新渲染")
    include_rotation: bool = Field(default=False, description="是否额外生成旋转 GIF；视觉索引只需要六视图")
    rotation_frames: int = Field(default=36, ge=12, le=120, description="旋转 GIF 帧数")


class RenderResponse(BaseModel):
    success: bool = Field(default=True, description="渲染是否成功")
    file_id: str | None = Field(default=None, description="被渲染的文件 ID")
    views: ViewImages = Field(description="六视图图片 URL；前端可直接拼接 API 域名显示")
    rotation_url: str | None = Field(default=None, description="旋转 GIF URL；未请求时为 null")
    cached: bool = Field(default=False, description="是否复用了已有缓存")


class ViewMatchRequest(BaseModel):
    query_file_id: str = Field(min_length=1, description="查询零件文件 ID，定义本次对照的六视图方向")
    candidate_file_id: str = Field(min_length=1, description="候选零件文件 ID，可来自不同零件库")
    method: Literal["rigid24", "hungarian"] = Field(default="rigid24", description="rigid24：24种右手三维旋转约束配对；hungarian：独立图像最优分配，不保证三维一致")
    force: bool = Field(default=False, description="强制重新渲染两件零件；旧版渲染缓存即使为false也会自动升级")


class ViewPair(BaseModel):
    query_view: str = Field(description="查询视图名称，前端的对照槽位")
    candidate_view: str = Field(description="该槽位对应的候选原始视图名称")
    rotation_degrees_ccw: int = Field(description="候选原始图片需要逆时针旋转的角度：0/90/180/270")
    score: float = Field(description="该对视图的轮廓及弱灰度匹配分数，范围0~1，不是准确率")


class ViewMatchResponse(BaseModel):
    status: str = Field(default="ready", description="ready表示完成配对")
    method: str = Field(description="实际使用的配对算法")
    query_file_id: str | None = Field(default=None, description="查询零件ID")
    candidate_file_id: str | None = Field(default=None, description="候选零件ID")
    score: float = Field(description="六对图像平均匹配分数，不替代检索综合分数")
    hungarian_score: float = Field(description="独立最优分配的参考上界；明显高于刚性分数时可能存在视图不一致")
    rotation_matrix: list[list[int]] | None = Field(description="候选标准坐标到查询标准坐标的3×3旋转矩阵；匈牙利模式为null")
    rigid_consistent: bool = Field(description="是否强制六张图对应同一个右手三维旋转")
    score_gap: float | None = Field(description="最优与次优刚性旋转分差；越小方向越不确定")
    ambiguous: bool = Field(description="方向是否存在歧义；对称件常为true，不代表失败")
    pairs: list[ViewPair] = Field(description="六视图一一配对及平面内旋转说明")
    query_views: ViewImages = Field(description="查询件标准六视图URL")
    candidate_views: ViewImages = Field(description="候选件配对前标准六视图URL")
    aligned_views: ViewImages = Field(description="已完成旋转的候选六视图URL，以查询视图名称为键；前端可直接按同名槽位显示，无需再旋转")
    warning: str = Field(description="方向歧义或非刚性配对提示")


class BuildIndexRequest(BaseModel):
    library_id: str = Field(default="default", description="需要建立索引的零件库 ID")
    directory: str | None = Field(default=None, description="可选服务器本地零件目录；通常留空使用上传目录")
    view_dir: str | None = Field(default=None, description="可选服务器本地六视图目录；通常留空使用渲染缓存")


class BuildIndexResponse(BaseModel):
    success: bool = Field(default=True, description="建库是否成功")
    library_id: str = Field(description="零件库 ID")
    library_name: str = Field(description="零件库名称")
    directory: str = Field(description="本次扫描的服务器目录")
    file_count: int = Field(description="扫描到的 STP/STEP 文件数量")
    index_count: int = Field(description="兼容字段，等于 text_index_count")
    text_index_count: int = Field(description="文本 Embedding 索引数量")
    geometric_index_count: int = Field(description="几何 FAISS 索引数量")
    visual_index_count: int = Field(description="视觉 FAISS 索引数量；渲染失败时可能小于文件数")
    elapsed_seconds: float = Field(description="建库耗时，单位秒")
    message: str = Field(description="建库结果摘要")


class SearchRequest(FileReference):
    include_view_alignment: bool = Field(default=True, description="是否为最终结果补充六视图配对；默认以查询件为标准自动返回aligned_views，无需单独调用/api/render/match")
    library_id: str | None = Field(default="default", description="旧版单零件库 ID；建议新前端使用 library_ids")
    library_ids: list[str] = Field(default_factory=list, description="目标零件库 ID 列表，支持一次选择多个独立零件库")
    result_limit: int | None = Field(default=None, ge=1, le=50, description="最终返回数量，对应前端数量选择器")
    final_top: int = Field(default=10, ge=1, le=50, description="旧版返回数量字段；未传 result_limit 时默认返回10条")
    coarse_top: int = Field(default=20, ge=1, le=200, description="精排前保留的候选数量")
    use_llm_rerank: bool = Field(default=True, description="是否调用大模型进行最终精排")
    use_vision: bool = Field(default=False, description="LLM 精排时是否使用六视图；默认关闭以避免逐件视觉分析，三路召回的视觉索引不受此字段控制")
    use_geometric: bool = Field(default=True, description="非三路模式下是否计算几何相似度")
    use_feature_extraction: bool = Field(default=False, description="是否在 LLM 精排前逐件提取结构化视觉特征；默认关闭")
    use_three_way: bool = Field(default=True, description="是否使用文本、几何、视觉三路融合召回；默认开启，返回 visual_similarity")
    use_3d_vision: bool = Field(default=False, description="是否在支持的多模态精排模式中使用旋转 GIF")
    auto_render: bool = Field(default=False, description="查询或候选缺少图片时是否尝试自动渲染；默认关闭，渲染应在入库阶段完成")
    api_key: str | None = Field(default=None, repr=False, description="可选 LLM 精排 API Key，覆盖服务器 LLM_API_KEY")
    base_url: str | None = Field(default=None, description="可选 LLM 精排服务地址，覆盖服务器 LLM_BASE_URL")
    llm_model: str | None = Field(default=None, description="可选 LLM 模型名，覆盖服务器配置")
    custom_query_text: str | None = Field(default=None, description="用户自定义检索条件，例如孔数、尺寸或形状要求")
    reference_file_ids: list[str] = Field(default_factory=list, description="可选参考图片文件 ID 列表")
    reference_dimensions: dict[str, float] | None = Field(default=None, description="可选参考尺寸，例如 length、width、height")
    save_report: bool = Field(default=False, description="是否生成 Word 检索报告")
    text_recall_k: int = Field(default=150, ge=1, le=500, description="三路模式的文本召回数量")
    geo_recall_k: int = Field(default=150, ge=1, le=500, description="三路模式的几何召回数量")
    visual_recall_k: int = Field(default=150, ge=1, le=500, description="三路模式的视觉召回数量")
    fusion_top_k: int = Field(default=20, ge=1, le=200, description="三路融合后进入精排的候选数量")

    @model_validator(mode="after")
    def apply_result_limit(self):
        if self.result_limit is not None:
            self.final_top = self.result_limit
        selected = self.library_ids or ([self.library_id] if self.library_id else [])
        normalized: list[str] = []
        seen: set[str] = set()
        for library_id in selected:
            value = str(library_id).strip()
            if value and value not in seen:
                normalized.append(value)
                seen.add(value)
        if not normalized:
            raise ValueError("library_id 和 library_ids 至少需要提供一个")
        if len(normalized) > 20:
            raise ValueError("一次检索最多选择20个零件库")
        self.library_ids = normalized
        self.library_id = normalized[0] if len(normalized) == 1 else None
        return self

    @property
    def effective_result_limit(self) -> int:
        return self.result_limit if self.result_limit is not None else self.final_top

    @property
    def effective_library_ids(self) -> list[str]:
        return list(self.library_ids)


class SearchResultItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int = Field(default=0, description="最终排名，从 1 开始")
    filename: str = Field(default="", description="候选零件文件名")
    filepath: str = Field(default="", description="候选零件服务器内部路径")
    similarity_score: float = Field(default=0.0, description="LLM 精排相似度；未启用 LLM 时通常为 0")
    embedding_similarity: float = Field(default=0.0, description="文本 Embedding 相似度")
    geometric_similarity: dict[str, Any] | None = Field(default=None, description="几何相似度及分项得分")
    hybrid_similarity: float = Field(default=0.0, description="召回和几何融合后的综合相似度")
    composite_score: float = Field(default=0.0, description="最终综合分数；结果按此字段降序排列，建议显示在结果卡片顶部")
    visual_similarity: float = Field(default=0.0, description="六视图视觉向量相似度")
    reason: str = Field(default="", description="LLM 或规则生成的匹配理由")
    key_matches: list[str] = Field(default_factory=list, description="关键匹配特征")
    file_id: str | None = Field(default=None, description="候选零件文件 ID，可用于详情和渲染接口")
    library_id: str | None = Field(default=None, description="候选零件所属零件库 ID")
    view_alignment: ViewMatchResponse | None = Field(default=None, description="候选相对于查询件的六视图配对结果")
    view_alignment_error: str | None = Field(default=None, description="视图配对失败原因；不影响原检索结果")


class SearchResponse(BaseModel):
    success: bool = Field(default=True, description="检索是否成功")
    library_id: str | None = Field(default=None, description="兼容字段；单库检索时为零件库 ID，多库检索时为 null")
    library_name: str | None = Field(default=None, description="兼容字段；单库检索时为零件库名称，多库检索时为 null")
    library_ids: list[str] = Field(description="本次检索使用的全部零件库 ID")
    library_names: list[str] = Field(description="本次检索使用的全部零件库名称，顺序与 library_ids 一致")
    query_file_id: str | None = Field(default=None, description="查询零件文件 ID")
    query_filename: str = Field(description="查询零件文件名")
    query_views: ViewImages = Field(default_factory=ViewImages, description="查询零件六视图URL；三路检索会在缺失时自动渲染")
    query_render_error: str | None = Field(default=None, description="查询件六视图渲染失败原因；成功时为null")
    query_metadata: PartMetadata | None = Field(default=None, description="本次重新解析的查询件几何与尺寸信息")
    elapsed_seconds: float = Field(description="检索耗时，单位秒")
    total_results: int = Field(description="实际返回结果数量")
    result_limit: int = Field(description="请求的最大返回数量")
    results: list[SearchResultItem] = Field(description="按 composite_score 综合分数降序排列的候选零件")
    report_url: str | None = Field(default=None, description="Word 报告下载地址；未生成时为 null")


class LTRTrainRequest(BaseModel):
    annotation_file_id: str | None = Field(default=None, description="训练标注 JSON 的上传文件 ID")
    annotation_path: str | None = Field(default=None, description="训练标注 JSON 的服务器路径")
    eval_annotation_file_id: str | None = Field(default=None, description="可选验证标注文件 ID")
    eval_annotation_path: str | None = Field(default=None, description="可选验证标注服务器路径")
    model_type: Literal["lightgbm", "mlp"] = Field(default="lightgbm", description="LTR 模型类型")
    model_name: str = Field(default="ltr_model.txt", description="保存的模型文件名")
    num_leaves: int = Field(default=31, ge=2, le=256, description="LightGBM 叶节点数量")
    learning_rate: float = Field(default=0.05, ge=0.001, le=1.0, description="训练学习率")
    num_iterations: int = Field(default=200, ge=10, le=5000, description="训练迭代次数")

    @model_validator(mode="after")
    def validate_annotation(self):
        if bool(self.annotation_file_id) == bool(self.annotation_path):
            raise ValueError("annotation_file_id 和 annotation_path 必须且只能提供一个")
        return self


class LTRTrainResponse(BaseModel):
    success: bool = Field(default=True, description="训练是否成功")
    model_type: str = Field(description="实际使用的模型类型")
    model_name: str = Field(description="保存的模型文件名")
    elapsed_seconds: float = Field(description="训练耗时，单位秒")
    history: dict[str, Any] = Field(description="训练和验证指标")


class LTREvaluateRequest(BaseModel):
    annotation_file_id: str | None = Field(default=None, description="评估标注 JSON 的上传文件 ID")
    annotation_path: str | None = Field(default=None, description="评估标注 JSON 的服务器路径")

    @model_validator(mode="after")
    def validate_annotation(self):
        if bool(self.annotation_file_id) == bool(self.annotation_path):
            raise ValueError("annotation_file_id 和 annotation_path 必须且只能提供一个")
        return self


class LTRPredictRequest(FileReference):
    candidate_file_ids: list[str] = Field(min_length=1, description="需要 LTR 重排的候选零件文件 ID")


class LTRPredictItem(BaseModel):
    file_id: str = Field(description="候选零件文件 ID")
    filename: str = Field(description="候选零件文件名")
    ltr_score: float = Field(description="LTR 模型预测得分")


class LTRPredictResponse(BaseModel):
    success: bool = Field(default=True, description="LTR 推理是否成功")
    results: list[LTRPredictItem] = Field(description="按 LTR 得分排序的候选零件")


class FeatureImportanceResponse(BaseModel):
    success: bool = Field(default=True, description="查询是否成功")
    feature_importance: dict[str, float] = Field(description="LTR 各输入特征的重要性")


class ErrorResponse(BaseModel):
    success: bool = Field(default=False, description="固定为 false")
    detail: str = Field(description="错误原因")
