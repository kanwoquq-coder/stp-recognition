from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import Settings
from app.errors import APIError
from app.schemas import (
    BatchUploadResponse,
    BuildIndexRequest,
    BuildIndexResponse,
    DeletePartResponse,
    FeatureImportanceResponse,
    FileCategory,
    FileItem,
    IngestionSummary,
    IndexSummary,
    InspectResponse,
    LibraryCreateRequest,
    LibraryItem,
    LTREvaluateRequest,
    LTRPredictItem,
    LTRPredictRequest,
    LTRPredictResponse,
    LTRTrainRequest,
    LTRTrainResponse,
    PartListResponse,
    PartRenderResult,
    RenderRequest,
    RenderResponse,
    ViewMatchRequest,
    ViewMatchResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    StatusResponse,
    UploadResponse,
)
from app.services import EngineService, LTRService, RenderService, STP_EXTENSIONS, to_builtin
from app.storage import DEFAULT_LIBRARY_ID, FileStore


def _candidate_file_id(store: FileStore, path: str) -> str | None:
    return store.file_id_for_path(path)


def _numeric_similarity(value: object) -> float | None:
    """将模型/索引返回的相似度安全转换为可排序浮点数。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip()
            is_percent = text.endswith("%")
            value = float(text.removesuffix("%"))
            if is_percent:
                value /= 100.0
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _first_similarity(result: dict, *fields: str) -> float:
    for field in fields:
        score = _numeric_similarity(result.get(field))
        if score is not None:
            return score
    return float("-inf")


def _search_composite_sort_key(result: dict) -> tuple[float, float, float]:
    """按综合相似度降序；LLM和文本相似度作为稳定的次级排序依据。"""
    composite_score = _first_similarity(
        result,
        "hybrid_similarity",
        "final_score",
        "fusion_score",
        "similarity_score",
        "embedding_similarity",
        "similarity",
    )
    llm_score = _first_similarity(result, "similarity_score")
    text_score = _first_similarity(result, "embedding_similarity", "similarity")
    return composite_score, llm_score, text_score


def _composite_similarity(result: dict) -> float:
    score = _search_composite_sort_key(result)[0]
    return 0.0 if score == float("-inf") else score


def _parse_part_metadata(store: FileStore, file_id: str) -> FileItem:
    record = store.record(file_id)
    if record.get("category") not in {"query", "library"}:
        return store.public_item(record)
    try:
        from stp_similarity import parse_stp_deep

        metadata = to_builtin(parse_stp_deep(record["path"]))
    except Exception as exc:
        metadata = {"parse_error": f"{type(exc).__name__}: {exc}"}
    return store.update_part_metadata(file_id, metadata)


async def _render_saved_part(
    store: FileStore,
    renders: RenderService,
    file_id: str,
    library_id: str | None,
    force: bool,
    include_rotation: bool,
    rotation_frames: int,
) -> PartRenderResult:
    record = store.record(file_id)
    try:
        views, rotation_url, cached = await asyncio.to_thread(
            renders.render,
            Path(record["path"]),
            force,
            include_rotation,
            rotation_frames,
            library_id,
        )
        return PartRenderResult(
            file_id=file_id,
            status="ready",
            views=views,
            rotation_url=rotation_url,
            cached=cached,
        )
    except APIError as exc:
        return PartRenderResult(file_id=file_id, status="failed", error=exc.detail)
    except Exception as exc:
        return PartRenderResult(
            file_id=file_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


async def _build_library(
    settings: Settings,
    store: FileStore,
    engines: EngineService,
    library_id: str,
    directory: str | None = None,
    view_dir: str | None = None,
) -> tuple[IndexSummary, Path, int, float]:
    library = store.library_item(library_id)
    resolved_directory = store.resolve_index_directory(directory, library_id)
    resolved_view_dir = None
    if view_dir:
        if not settings.allow_local_paths:
            raise APIError(403, "本地路径模式已关闭，不能指定 view_dir")
        resolved_view_dir = Path(view_dir).expanduser().resolve()
        if not resolved_view_dir.is_dir():
            raise APIError(404, f"视图目录不存在: {resolved_view_dir}")

    store.set_library_index_state(library_id, "building")
    start = time.perf_counter()
    try:
        counts = await asyncio.to_thread(
            engines.build_index,
            library_id,
            resolved_directory,
            resolved_view_dir,
        )
        summary = store.set_library_index_state(
            library_id,
            "ready",
            text_count=counts["text"],
            geometric_count=counts["geometric"],
            visual_count=counts["visual"],
        )
    except APIError as exc:
        store.set_library_index_state(library_id, "failed", error=exc.detail)
        raise
    except Exception as exc:
        detail = f"三路索引构建失败: {type(exc).__name__}: {exc}"
        store.set_library_index_state(library_id, "failed", error=detail)
        raise APIError(500, detail) from exc

    elapsed = time.perf_counter() - start
    file_count = sum(
        1
        for path in resolved_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in STP_EXTENSIONS
    )
    # Keep the lookup above so a concurrently deleted library fails clearly.
    _ = library
    return summary, resolved_directory, file_count, elapsed


def create_router(
    settings: Settings,
    store: FileStore,
    engines: EngineService,
    renders: RenderService,
    ltr: LTRService,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/status", response_model=StatusResponse, tags=["1. 健康检查"])
    async def status() -> StatusResponse:
        capabilities = await asyncio.to_thread(engines.capabilities)
        libraries = await asyncio.to_thread(store.list_libraries)
        return StatusResponse(
            version="2.2.0",
            port=settings.port,
            engine_initialized=engines.initialized_count() > 0,
            index_count=sum(item.index.text_count for item in libraries),
            has_api_key=bool(settings.api_key),
            has_faiss=capabilities["has_faiss"],
            has_render=capabilities["has_render"],
            has_ltr=await asyncio.to_thread(ltr.capability),
            ltr_loaded=ltr.loaded,
            allow_local_paths=settings.allow_local_paths,
            library_count=len(libraries),
            total_parts=sum(item.part_count for item in libraries),
        )

    @router.post(
        "/libraries",
        response_model=LibraryItem,
        tags=["2. 零件库"],
        status_code=201,
    )
    async def create_library(request: LibraryCreateRequest) -> LibraryItem:
        return await asyncio.to_thread(
            store.create_library, request.name, request.description
        )

    @router.get("/libraries", response_model=list[LibraryItem], tags=["2. 零件库"])
    async def list_libraries() -> list[LibraryItem]:
        return await asyncio.to_thread(store.list_libraries)

    @router.get(
        "/libraries/{library_id}",
        response_model=LibraryItem,
        tags=["2. 零件库"],
    )
    async def get_library(library_id: str) -> LibraryItem:
        return await asyncio.to_thread(store.library_item, library_id)

    @router.post("/files/upload", response_model=UploadResponse, tags=["3. 文件上传"])
    async def upload_file(
        file: Annotated[UploadFile, File(description="STP、参考图片或 LTR 标注 JSON")],
        category: Annotated[FileCategory, Form(description="文件用途：query、library、reference 或 annotation")] = "query",
        library_id: Annotated[str | None, Form(description="category=library 时指定目标零件库 ID")] = None,
        auto_render: Annotated[bool, Form(description="是否在保存后立即生成六视图")] = False,
        render_force: Annotated[bool, Form(description="是否强制覆盖已有六视图缓存")] = False,
        include_rotation: Annotated[bool, Form(description="是否额外生成旋转 GIF；视觉索引不要求 GIF")] = False,
        rotation_frames: Annotated[int, Form(ge=12, le=120, description="旋转 GIF 帧数")] = 36,
        auto_index: Annotated[bool, Form(description="category=library 时是否在渲染后立即重建三路索引")] = False,
    ) -> UploadResponse:
        started = time.perf_counter()
        if auto_index and category != "library":
            raise APIError(422, "auto_index 仅适用于 category=library")
        if auto_render and category not in {"query", "library"}:
            raise APIError(422, "auto_render 仅适用于 STP/STEP 文件")
        assigned_library = (
            library_id or DEFAULT_LIBRARY_ID if category == "library" else None
        )
        saved = await store.save(file, category, assigned_library)
        if category in {"query", "library"}:
            saved = await asyncio.to_thread(_parse_part_metadata, store, saved.file_id)

        render_result = None
        if auto_render:
            render_result = await _render_saved_part(
                store,
                renders,
                saved.file_id,
                assigned_library,
                render_force,
                include_rotation,
                rotation_frames,
            )

        summary = (
            store.index_summary(assigned_library) if assigned_library else None
        )
        index_error = None
        if auto_index and assigned_library:
            try:
                summary, _, _, _ = await _build_library(
                    settings, store, engines, assigned_library
                )
            except APIError as exc:
                summary = store.index_summary(assigned_library)
                index_error = exc.detail
            saved = store.public_item(store.record(saved.file_id))
        parsed_count = int(bool(saved.metadata and not saved.metadata.parse_error))
        rendered_count = int(bool(render_result and render_result.status == "ready"))
        render_failed_count = int(bool(render_result and render_result.status == "failed"))
        processing = IngestionSummary(
            uploaded_count=1,
            metadata_parsed_count=parsed_count,
            rendered_count=rendered_count,
            render_failed_count=render_failed_count,
            text_index_count=summary.text_count if summary else 0,
            geometric_index_count=summary.geometric_count if summary else 0,
            visual_index_count=summary.visual_count if summary else 0,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            complete=(
                (not auto_render or rendered_count == 1)
                and (
                    not auto_index
                    or bool(
                        summary
                        and summary.status == "ready"
                        and (not auto_render or summary.visual_count >= summary.text_count)
                    )
                )
            ),
        )
        return UploadResponse(
            file=saved,
            render=render_result,
            index=summary,
            index_error=index_error,
            processing=processing,
        )

    @router.post(
        "/files/upload-batch",
        response_model=BatchUploadResponse,
        tags=["3. 文件上传"],
    )
    async def upload_batch(
        files: Annotated[list[UploadFile], File(description="批量上传文件")],
        category: Annotated[FileCategory, Form(description="批量建库时使用 library")] = "library",
        library_id: Annotated[str | None, Form(description="目标零件库 ID；不传时使用 default")] = None,
        auto_render: Annotated[bool, Form(description="是否在建索引前为每个零件生成六视图")] = True,
        render_force: Annotated[bool, Form(description="是否强制覆盖已有六视图缓存")] = False,
        include_rotation: Annotated[bool, Form(description="是否额外为每个零件生成旋转 GIF")] = False,
        rotation_frames: Annotated[int, Form(ge=12, le=120, description="旋转 GIF 帧数")] = 36,
        auto_index: Annotated[bool, Form(description="是否在渲染后立即重建文本、几何、视觉三路索引")] = False,
    ) -> BatchUploadResponse:
        started = time.perf_counter()
        if len(files) > 500:
            raise APIError(413, "单次最多上传 500 个文件")
        if auto_index and category != "library":
            raise APIError(422, "auto_index 仅适用于 category=library")
        if auto_render and category not in {"query", "library"}:
            raise APIError(422, "auto_render 仅适用于 STP/STEP 文件")
        assigned_library = (
            library_id or DEFAULT_LIBRARY_ID if category == "library" else None
        )
        saved: list[FileItem] = []
        for upload in files:
            item = await store.save(upload, category, assigned_library)
            if category in {"query", "library"}:
                item = await asyncio.to_thread(
                    _parse_part_metadata, store, item.file_id
                )
            saved.append(item)

        render_results: list[PartRenderResult] = []
        if auto_render:
            for item in saved:
                render_results.append(
                    await _render_saved_part(
                        store,
                        renders,
                        item.file_id,
                        assigned_library,
                        render_force,
                        include_rotation,
                        rotation_frames,
                    )
                )

        summary = (
            store.index_summary(assigned_library) if assigned_library else None
        )
        index_error = None
        if auto_index and assigned_library:
            try:
                summary, _, _, _ = await _build_library(
                    settings, store, engines, assigned_library
                )
            except APIError as exc:
                summary = store.index_summary(assigned_library)
                index_error = exc.detail
            saved = [
                store.public_item(store.record(item.file_id)) for item in saved
            ]
        return BatchUploadResponse(
            files=saved,
            library_id=assigned_library,
            renders=render_results,
            index=summary,
            index_error=index_error,
            processing=IngestionSummary(
                uploaded_count=len(saved),
                metadata_parsed_count=sum(
                    bool(item.metadata and not item.metadata.parse_error) for item in saved
                ),
                rendered_count=sum(item.status == "ready" for item in render_results),
                render_failed_count=sum(item.status == "failed" for item in render_results),
                text_index_count=summary.text_count if summary else 0,
                geometric_index_count=summary.geometric_count if summary else 0,
                visual_index_count=summary.visual_count if summary else 0,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                complete=(
                    (not auto_render or all(item.status == "ready" for item in render_results))
                    and (
                        not auto_index
                        or bool(
                            summary
                            and summary.status == "ready"
                            and (not auto_render or summary.visual_count >= summary.text_count)
                        )
                    )
                ),
            ),
        )

    @router.get("/files", response_model=list[FileItem], tags=["4. 零件管理"])
    async def list_files(
        category: FileCategory | None = Query(default=None),
        library_id: str | None = Query(default=None),
    ) -> list[FileItem]:
        return await asyncio.to_thread(store.list, category, library_id)

    @router.get("/parts", response_model=PartListResponse, tags=["4. 零件管理"])
    async def list_parts(
        library_id: str | None = Query(default=None),
        keyword: str | None = Query(default=None, max_length=100),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> PartListResponse:
        return await asyncio.to_thread(
            store.list_parts, library_id, keyword, offset, limit
        )

    @router.get(
        "/parts/{file_id}",
        response_model=FileItem,
        tags=["4. 零件管理"],
    )
    async def get_part(file_id: str) -> FileItem:
        record = await asyncio.to_thread(store.record, file_id)
        if record.get("category") != "library":
            raise APIError(404, f"零件库文件不存在: {file_id}")
        return store.public_item(record)

    @router.delete(
        "/parts/{file_id}",
        response_model=DeletePartResponse,
        tags=["4. 零件管理"],
        summary="删除零件并同步清理三路索引和渲染缓存",
    )
    async def delete_part(file_id: str) -> DeletePartResponse:
        record = await asyncio.to_thread(store.record, file_id)
        if record.get("category") != "library" or not record.get("library_id"):
            raise APIError(404, f"零件库文件不存在: {file_id}")

        library_id = record["library_id"]
        original_name = record["original_name"]
        stp_path = Path(record["path"])
        current_index = store.index_summary(library_id)
        index_removed = {"text": False, "geometric": False, "visual": False}

        # 搜索、建库和删除共用同一把操作锁，避免检索读到半删除状态。
        def delete_synchronously():
            nonlocal index_removed
            with engines.operation_lock:
                if current_index.status == "ready":
                    index_removed, counts = engines.remove_part_from_index(
                        library_id, stp_path
                    )
                else:
                    counts = {"text": 0, "geometric": 0, "visual": 0}
                render_deleted = renders.delete_cache(stp_path, library_id)
                _, file_deleted = store.delete(file_id)
                remaining = store.library_item(library_id).part_count
                if remaining == 0 or current_index.status != "ready":
                    summary = store.clear_library_index_state(library_id)
                else:
                    summary = store.set_library_index_state(
                        library_id,
                        "ready",
                        text_count=counts["text"],
                        geometric_count=counts["geometric"],
                        visual_count=counts["visual"],
                    )
                return file_deleted, render_deleted, remaining, summary

        file_deleted, render_deleted, remaining, summary = await asyncio.to_thread(
            delete_synchronously
        )
        return DeletePartResponse(
            file_id=file_id,
            original_name=original_name,
            library_id=library_id,
            file_deleted=file_deleted,
            render_deleted=render_deleted,
            index_removed=index_removed,
            index=summary,
            remaining_parts=remaining,
            message=f"零件 {original_name} 已删除，零件库剩余 {remaining} 个零件",
        )

    @router.get("/files/{file_id}/download", tags=["4. 零件管理"])
    async def download_file(file_id: str) -> FileResponse:
        record = await asyncio.to_thread(store.record, file_id)
        return FileResponse(
            record["path"],
            media_type=record["content_type"],
            filename=record["original_name"],
        )

    @router.get(
        "/files/{file_id}/inspect",
        response_model=InspectResponse,
        tags=["4. 零件管理"],
    )
    async def inspect_file(file_id: str) -> InspectResponse:
        record = await asyncio.to_thread(store.record, file_id)
        metadata = record.get("metadata")
        if metadata is None:
            item = await asyncio.to_thread(_parse_part_metadata, store, file_id)
            metadata = item.metadata
        return InspectResponse(file_id=file_id, metadata=metadata or {})

    @router.post("/render", response_model=RenderResponse, tags=["5. 在线渲染"])
    async def render_file(request: RenderRequest) -> RenderResponse:
        library_id = None
        if request.file_id:
            library_id = store.record(request.file_id).get("library_id")
        path = store.resolve_file(request.file_id, request.file_path, STP_EXTENSIONS)
        views, rotation_url, cached = await asyncio.to_thread(
            renders.render,
            path,
            request.force,
            request.include_rotation,
            request.rotation_frames,
            library_id,
        )
        return RenderResponse(
            file_id=request.file_id,
            views=views,
            rotation_url=rotation_url,
            cached=cached,
        )

    @router.post("/render/match", response_model=ViewMatchResponse, tags=["5. 在线渲染"])
    async def match_render_views(request: ViewMatchRequest) -> ViewMatchResponse:
        query = store.resolve_file(request.query_file_id, None, STP_EXTENSIONS)
        candidate = store.resolve_file(request.candidate_file_id, None, STP_EXTENSIONS)
        result = await asyncio.to_thread(
            renders.match, query, candidate,
            store.record(request.query_file_id).get("library_id"),
            store.record(request.candidate_file_id).get("library_id"),
            request.method, request.force,
        )
        return ViewMatchResponse(**result, query_file_id=request.query_file_id,
                                 candidate_file_id=request.candidate_file_id)

    async def build_response(request: BuildIndexRequest) -> BuildIndexResponse:
        library = store.library_item(request.library_id)
        summary, directory, file_count, elapsed = await _build_library(
            settings,
            store,
            engines,
            request.library_id,
            request.directory,
            request.view_dir,
        )
        return BuildIndexResponse(
            library_id=request.library_id,
            library_name=library.name,
            directory=str(directory),
            file_count=file_count,
            index_count=summary.text_count,
            text_index_count=summary.text_count,
            geometric_index_count=summary.geometric_count,
            visual_index_count=summary.visual_count,
            elapsed_seconds=round(elapsed, 3),
            message=(
                f"三路索引构建完成：文本 {summary.text_count}，"
                f"几何 {summary.geometric_count}，视觉 {summary.visual_count}"
            ),
        )

    @router.post("/build-index", response_model=BuildIndexResponse, tags=["6. 索引"])
    async def build_index(request: BuildIndexRequest) -> BuildIndexResponse:
        return await build_response(request)

    @router.post(
        "/libraries/{library_id}/build-index",
        response_model=BuildIndexResponse,
        tags=["6. 索引"],
    )
    async def build_library_index(
        library_id: str,
        request: BuildIndexRequest,
    ) -> BuildIndexResponse:
        request.library_id = library_id
        return await build_response(request)

    @router.post("/search", response_model=SearchResponse, tags=["7. 相似检索"])
    async def search(request: SearchRequest) -> SearchResponse:
        library_ids = request.effective_library_ids
        libraries = [store.library_item(library_id) for library_id in library_ids]
        unavailable = [
            f"{library.name}({library.index.status})"
            for library in libraries
            if library.index.status != "ready"
        ]
        if unavailable:
            raise APIError(
                409,
                "以下零件库尚未完成三路索引，请先建库：" + "、".join(unavailable),
            )
        if len(library_ids) > 1 and request.save_report:
            raise APIError(422, "多零件库检索暂不支持合并Word报告，请设置 save_report=false")
        query_path = store.resolve_file(
            request.file_id, request.file_path, STP_EXTENSIONS
        )
        query_library_id = None
        if request.file_id:
            query_record = store.record(request.file_id)
            query_library_id = query_record.get("library_id")
            query_item = await asyncio.to_thread(_parse_part_metadata, store, request.file_id)
            query_metadata = query_item.metadata
        else:
            from stp_similarity import parse_stp_deep
            query_metadata = to_builtin(await asyncio.to_thread(parse_stp_deep, str(query_path)))

        metadata_error = (
            query_metadata.parse_error
            if hasattr(query_metadata, "parse_error")
            else (query_metadata or {}).get("parse_error")
        )
        if metadata_error:
            raise APIError(422, f"查询文件不是可检索的有效STEP文件：{metadata_error}")

        query_views: dict[str, str] = {}
        query_view_paths: list[str] | None = None
        query_render_error = None
        # Visual recall cannot run without a query vector. Ensure the query has
        # canonical views before entering one or several library engines.
        if request.use_three_way or request.use_vision:
            try:
                query_views, _, _ = await asyncio.to_thread(
                    renders.render, query_path, False, False, 36, query_library_id
                )
                query_view_paths = renders.view_paths(query_path, query_library_id)
            except Exception as exc:
                query_render_error = getattr(exc, "detail", f"{type(exc).__name__}: {exc}")
        reference_images = [
            str(store.resolve_file(file_id, None, {".png", ".jpg", ".jpeg", ".webp"}))
            for file_id in request.reference_file_ids
        ]
        start = time.perf_counter()
        raw_results: list[dict] = []
        library_name_by_id = {
            library.library_id: library.name for library in libraries
        }
        for library_id in library_ids:
            library_results = await asyncio.to_thread(
                engines.search,
                library_id,
                query_path,
                request,
                reference_images,
                query_view_paths,
            )
            for result in library_results:
                result["_source_library_id"] = library_id
                result["_source_library_name"] = library_name_by_id[library_id]
            raw_results.extend(library_results)
        elapsed = time.perf_counter() - start

        query_resolved = query_path.resolve()
        filtered_results = []
        # 大模型返回的 rankings 顺序并不总是与分数字段一致。先按检索阶段
        # 计算的综合相似度排序，再过滤查询件和截断数量，避免高综合分候选
        # 落到后面，甚至在 result_limit 截断时被丢弃。
        sorted_results = sorted(
            raw_results,
            key=_search_composite_sort_key,
            reverse=True,
        )
        seen_paths: set[str] = set()
        for result in sorted_results:
            filepath = result.get("filepath", result.get("path", ""))
            try:
                if filepath:
                    resolved = str(Path(filepath).resolve())
                    if Path(resolved) == query_resolved or resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
            except (OSError, ValueError):
                pass
            filtered_results.append(result)
            if len(filtered_results) >= request.effective_result_limit:
                break

        items: list[SearchResultItem] = []
        for index, result in enumerate(filtered_results, 1):
            filepath = result.get("filepath", result.get("path", ""))
            geo = result.get("geometric_similarity")
            if not isinstance(geo, dict):
                geo = None
            candidate_id = _candidate_file_id(store, filepath) if filepath else None
            composite_score = _composite_similarity(result)
            item = {
                **result,
                "rank": index,
                "filename": result.get("filename", Path(filepath).name),
                "filepath": filepath,
                "geometric_similarity": geo,
                # 两个字段数值相同：hybrid_similarity 保持旧前端兼容；
                # composite_score 是语义明确的新字段，供卡片顶部展示。
                "hybrid_similarity": composite_score,
                "composite_score": composite_score,
                "file_id": candidate_id,
                "library_id": result.get("_source_library_id"),
                "library_name": result.get("_source_library_name"),
            }
            if request.include_view_alignment and filepath:
                try:
                    query_library = store.record(request.file_id).get("library_id") if request.file_id else None
                    alignment = await asyncio.to_thread(
                        renders.match, query_path, Path(filepath), query_library,
                        result.get("_source_library_id"),
                    )
                    item["view_alignment"] = {**alignment, "query_file_id": request.file_id,
                                              "candidate_file_id": candidate_id}
                except Exception as exc:
                    item["view_alignment_error"] = getattr(exc, "detail", str(exc))
            items.append(SearchResultItem.model_validate(item))

        report_url = None
        if request.save_report:
            report_path = settings.report_dir / (
                f"{library_ids[0]}_{query_path.stem}_search_report.docx"
            )
            if report_path.exists():
                report_url = f"/api/reports/{quote(report_path.name)}"
        return SearchResponse(
            library_id=library_ids[0] if len(library_ids) == 1 else None,
            library_name=libraries[0].name if len(libraries) == 1 else None,
            library_ids=library_ids,
            library_names=[library.name for library in libraries],
            query_file_id=request.file_id,
            query_filename=query_path.name,
            query_views=query_views,
            query_render_error=query_render_error,
            query_metadata=query_metadata,
            elapsed_seconds=round(time.perf_counter() - start, 3),
            total_results=len(items),
            result_limit=request.effective_result_limit,
            results=items,
            report_url=report_url,
        )

    @router.get("/reports/{filename}", tags=["7. 相似检索"])
    async def download_report(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        report_path = (settings.report_dir / safe_name).resolve()
        if report_path.parent != settings.report_dir.resolve() or not report_path.is_file():
            raise APIError(404, "报告不存在")
        return FileResponse(report_path, filename=safe_name)

    @router.post("/ltr/train", response_model=LTRTrainResponse, tags=["8. LTR"])
    async def train_ltr(request: LTRTrainRequest) -> LTRTrainResponse:
        annotation = store.resolve_annotation(
            request.annotation_file_id, request.annotation_path
        )
        evaluation = None
        if request.eval_annotation_file_id or request.eval_annotation_path:
            evaluation = store.resolve_annotation(
                request.eval_annotation_file_id, request.eval_annotation_path
            )
        start = time.perf_counter()
        model_name, history = await asyncio.to_thread(
            ltr.train, request, annotation, evaluation
        )
        return LTRTrainResponse(
            model_type=ltr._pipeline.model_type,
            model_name=model_name,
            elapsed_seconds=round(time.perf_counter() - start, 3),
            history=history,
        )

    @router.post("/ltr/evaluate", tags=["8. LTR"])
    async def evaluate_ltr(request: LTREvaluateRequest) -> dict:
        annotation = store.resolve_annotation(
            request.annotation_file_id, request.annotation_path
        )
        results = await asyncio.to_thread(ltr.evaluate, annotation)
        return {"success": True, **results}

    @router.get(
        "/ltr/feature-importance",
        response_model=FeatureImportanceResponse,
        tags=["8. LTR"],
    )
    async def feature_importance() -> FeatureImportanceResponse:
        result = await asyncio.to_thread(ltr.feature_importance)
        return FeatureImportanceResponse(feature_importance=result)

    @router.post("/ltr/predict", response_model=LTRPredictResponse, tags=["8. LTR"])
    async def predict_ltr(request: LTRPredictRequest) -> LTRPredictResponse:
        query_path = store.resolve_file(
            request.file_id, request.file_path, STP_EXTENSIONS
        )
        candidates = [
            (file_id, store.resolve_file(file_id, None, STP_EXTENSIONS))
            for file_id in request.candidate_file_ids
        ]
        result = await asyncio.to_thread(ltr.predict, query_path, candidates)
        return LTRPredictResponse(
            results=[LTRPredictItem.model_validate(item) for item in result]
        )

    return router
