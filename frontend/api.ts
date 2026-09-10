export const API_BASE =
  (globalThis as any).__STP_API_BASE__ ?? "http://localhost:8001";

export type FileCategory = "query" | "library" | "reference" | "annotation";
export type IndexStatus = "not_built" | "building" | "ready" | "failed";

export interface IndexSummary {
  status: IndexStatus;
  text_count: number;
  geometric_count: number;
  visual_count: number;
  indexed_at: string | null;
  error: string | null;
}

export interface PartLibrary {
  library_id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  part_count: number;
  indexed_part_count: number;
  index: IndexSummary;
}

export interface StoredFile {
  file_id: string;
  original_name: string;
  category: FileCategory;
  size: number;
  content_type: string;
  created_at: string;
  download_url: string;
  library_id: string | null;
  library_name: string | null;
  index_status: IndexStatus;
  indexed_at: string | null;
  metadata: Record<string, any> | null;
}

/** 六视图 URL；前端使用 mediaUrl() 转为完整地址。 */
export interface ViewImages {
  front: string | null;  // 主视图/前视图
  back: string | null;   // 后视图
  top: string | null;    // 俯视图/顶视图
  bottom: string | null; // 仰视图/底视图
  left: string | null;   // 左视图
  right: string | null;  // 右视图
}

export interface PartRenderResult {
  file_id: string;
  status: "ready" | "skipped" | "failed";
  views: ViewImages;
  rotation_url: string | null;
  cached: boolean;
  error: string | null;
}

export interface IngestionSummary {
  uploaded_count: number;
  metadata_parsed_count: number;
  rendered_count: number;
  render_failed_count: number;
  text_index_count: number;
  geometric_index_count: number;
  visual_index_count: number;
  elapsed_seconds: number;
  complete: boolean;
}

export interface IngestionOptions {
  autoIndex?: boolean;
  autoRender?: boolean;
  renderForce?: boolean;
  includeRotation?: boolean;
  rotationFrames?: number;
}

export interface UploadWithIndexResult {
  success: true;
  files: StoredFile[];
  library_id: string;
  renders: PartRenderResult[];
  index: IndexSummary | null;
  index_error: string | null;
  processing: IngestionSummary;
}

export interface PartPage {
  success: true;
  library_id: string | null;
  total: number;
  offset: number;
  limit: number;
  items: StoredFile[];
}

export interface DeletePartResult {
  success: true;
  file_id: string;
  original_name: string;
  library_id: string;
  file_deleted: boolean;
  render_deleted: boolean;
  index_removed: {
    text: boolean;
    geometric: boolean;
    visual: boolean;
  };
  index: IndexSummary;
  remaining_parts: number;
  message: string;
}

export interface RenderResult {
  success: true;
  file_id: string;
  views: ViewImages;
  rotation_url: string | null;
  cached: boolean;
}

export interface SearchOptions {
  library_id?: string;
  /** 新版多库检索字段；非空时优先于 library_id。 */
  library_ids?: string[];
  result_limit?: number;
  /** @deprecated Prefer result_limit. */
  final_top?: number;
  coarse_top?: number;
  use_llm_rerank?: boolean;
  use_vision?: boolean;
  use_geometric?: boolean;
  use_feature_extraction?: boolean;
  use_three_way?: boolean;
  use_3d_vision?: boolean;
  auto_render?: boolean;
  custom_query_text?: string;
  reference_file_ids?: string[];
  reference_dimensions?: Record<string, number>;
  save_report?: boolean;
  text_recall_k?: number;
  geo_recall_k?: number;
  visual_recall_k?: number;
  fusion_top_k?: number;
}

export interface SearchResultItem {
  rank: number;
  filename: string;
  file_id: string | null;
  /** 最终综合分数；后端按此字段降序，结果卡片顶部应显示此值。 */
  composite_score: number;
  /** composite_score 的兼容字段。 */
  hybrid_similarity: number;
  similarity_score: number;
  embedding_similarity: number;
  geometric_similarity: Record<string, any> | null;
  visual_similarity: number;
  reason: string;
  [key: string]: any;
}

export interface SearchResponse {
  success: true;
  library_id: string | null;
  library_name: string | null;
  library_ids: string[];
  library_names: string[];
  query_file_id: string | null;
  query_filename: string;
  elapsed_seconds: number;
  total_results: number;
  result_limit: number;
  results: SearchResultItem[];
  report_url: string | null;
}

export class StpApiError extends Error {
  constructor(public status: number, public body: any) {
    super(body?.detail ?? `HTTP ${status}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new StpApiError(response.status, body);
  return body as T;
}

export function mediaUrl(relativeUrl: string): string {
  return relativeUrl.startsWith("http") ? relativeUrl : `${API_BASE}${relativeUrl}`;
}

function appendIngestionOptions(body: FormData, options: IngestionOptions) {
  body.append("auto_index", String(options.autoIndex ?? false));
  body.append("auto_render", String(options.autoRender ?? true));
  body.append("render_force", String(options.renderForce ?? false));
  body.append("include_rotation", String(options.includeRotation ?? false));
  body.append("rotation_frames", String(options.rotationFrames ?? 36));
}

function normalizeRenderResult<T extends PartRenderResult>(result: T): T {
  result.views = Object.fromEntries(
    Object.entries(result.views).map(([name, url]) => [
      name,
      url ? mediaUrl(url) : null,
    ]),
  ) as unknown as ViewImages;
  if (result.rotation_url) result.rotation_url = mediaUrl(result.rotation_url);
  return result;
}

export async function getStatus() {
  return request<any>("/api/status");
}

export async function createLibrary(name: string, description = "") {
  return request<PartLibrary>("/api/libraries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
}

export async function listLibraries() {
  return request<PartLibrary[]>("/api/libraries");
}

export async function getLibrary(libraryId: string) {
  return request<PartLibrary>(
    `/api/libraries/${encodeURIComponent(libraryId)}`,
  );
}

export async function uploadFile(
  file: File,
  category: FileCategory = "query",
  options: { libraryId?: string } & IngestionOptions = {},
) {
  const body = new FormData();
  body.append("file", file);
  body.append("category", category);
  if (options.libraryId) body.append("library_id", options.libraryId);
  appendIngestionOptions(body, { ...options, autoRender: options.autoRender ?? false });
  const result = await request<{
    success: true;
    file: StoredFile;
    index: IndexSummary | null;
    index_error: string | null;
  }>("/api/files/upload", { method: "POST", body });
  return result.file;
}

export async function uploadFiles(
  files: File[],
  category: FileCategory = "library",
  options: { libraryId?: string } & IngestionOptions = {},
) {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  body.append("category", category);
  if (options.libraryId) body.append("library_id", options.libraryId);
  appendIngestionOptions(body, options);
  const result = await request<{
    success: true;
    files: StoredFile[];
    library_id: string | null;
    renders: PartRenderResult[];
    index: IndexSummary | null;
    index_error: string | null;
    processing: IngestionSummary;
  }>("/api/files/upload-batch", { method: "POST", body });
  return result.files;
}

/** 推荐入库流程：上传 → 六视图渲染 → 一次性建立文本/几何/视觉三路索引。 */
export async function uploadLibraryParts(
  files: File[],
  libraryId: string,
  options: boolean | IngestionOptions = true,
) {
  const config: IngestionOptions =
    typeof options === "boolean" ? { autoIndex: options } : options;
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  body.append("category", "library");
  body.append("library_id", libraryId);
  appendIngestionOptions(body, {
    autoIndex: config.autoIndex ?? true,
    autoRender: config.autoRender ?? true,
    renderForce: config.renderForce ?? false,
    includeRotation: config.includeRotation ?? false,
    rotationFrames: config.rotationFrames ?? 36,
  });
  const result = await request<UploadWithIndexResult>("/api/files/upload-batch", {
    method: "POST",
    body,
  });
  result.renders = result.renders.map(normalizeRenderResult);
  return result;
}

export async function listFiles(
  category?: FileCategory,
  libraryId?: string,
) {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (libraryId) params.set("library_id", libraryId);
  const query = params.size ? `?${params.toString()}` : "";
  return request<StoredFile[]>(`/api/files${query}`);
}

export async function listParts(options: {
  libraryId?: string;
  keyword?: string;
  offset?: number;
  limit?: number;
} = {}) {
  const params = new URLSearchParams();
  if (options.libraryId) params.set("library_id", options.libraryId);
  if (options.keyword) params.set("keyword", options.keyword);
  params.set("offset", String(options.offset ?? 0));
  params.set("limit", String(options.limit ?? 50));
  return request<PartPage>(`/api/parts?${params.toString()}`);
}

export async function getPart(fileId: string) {
  return request<StoredFile>(`/api/parts/${encodeURIComponent(fileId)}`);
}

/** 删除零件，并同步清理原文件、渲染缓存和三路索引。 */
export async function deletePart(fileId: string) {
  return request<DeletePartResult>(
    `/api/parts/${encodeURIComponent(fileId)}`,
    { method: "DELETE" },
  );
}

export async function inspectFile(fileId: string) {
  return request<any>(`/api/files/${encodeURIComponent(fileId)}/inspect`);
}

export async function renderFile(
  fileId: string,
  options: {
    force?: boolean;
    includeRotation?: boolean;
    rotationFrames?: number;
  } = {},
) {
  const result = await request<RenderResult>("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_id: fileId,
      force: options.force ?? false,
      include_rotation: options.includeRotation ?? false,
      rotation_frames: options.rotationFrames ?? 36,
    }),
  });
  result.views = Object.fromEntries(
    Object.entries(result.views).map(([name, url]) => [
      name,
      url ? mediaUrl(url) : null,
    ]),
  ) as unknown as ViewImages;
  if (result.rotation_url) result.rotation_url = mediaUrl(result.rotation_url);
  return result;
}

export async function buildIndex(
  libraryId = "default",
  directory?: string,
) {
  return request<any>("/api/build-index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      library_id: libraryId,
      directory: directory ?? null,
    }),
  });
}

export async function searchSimilar(
  fileId: string,
  options: SearchOptions = {},
) {
  const response = await request<SearchResponse>("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // 先展开可选项，再写入规范化字段，避免首次渲染时显式传入的
      // undefined 覆盖默认数量，导致后端回退到旧版的5条默认值。
      ...options,
      file_id: fileId,
      library_id: options.library_ids?.length
        ? undefined
        : (options.library_id ?? "default"),
      library_ids: options.library_ids?.length
        ? options.library_ids
        : undefined,
      result_limit: options.result_limit ?? options.final_top ?? 10,
      use_llm_rerank: options.use_llm_rerank ?? true,
      use_vision: options.use_vision ?? false,
      use_feature_extraction: options.use_feature_extraction ?? false,
      use_3d_vision: options.use_3d_vision ?? false,
      auto_render: options.auto_render ?? false,
      fusion_top_k: options.fusion_top_k ?? 20,
    }),
  });
  // 防止代理缓存或旧后端返回顺序异常；页面始终按综合分数展示。
  response.results.forEach((item) => {
    const score = Number(item.composite_score ?? item.hybrid_similarity ?? 0);
    item.composite_score = Number.isFinite(score) ? score : 0;
  });
  response.results.sort(
    (left, right) => right.composite_score - left.composite_score,
  );
  return response;
}

export async function trainLtr(
  annotationFileId: string,
  modelType: "lightgbm" | "mlp" = "lightgbm",
) {
  return request<any>("/api/ltr/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      annotation_file_id: annotationFileId,
      model_type: modelType,
    }),
  });
}

export async function evaluateLtr(annotationFileId: string) {
  return request<any>("/api/ltr/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ annotation_file_id: annotationFileId }),
  });
}

export async function getLtrFeatureImportance() {
  return request<any>("/api/ltr/feature-importance");
}

export async function predictLtr(
  queryFileId: string,
  candidateFileIds: string[],
) {
  return request<any>("/api/ltr/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_id: queryFileId,
      candidate_file_ids: candidateFileIds,
    }),
  });
}
