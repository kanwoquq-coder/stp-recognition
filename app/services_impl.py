from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.config import Settings
from app.errors import APIError, CapabilityUnavailableError


STP_EXTENSIONS = {".stp", ".step"}


def to_builtin(value: Any) -> Any:
    """Convert numpy values and non-finite floats into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_builtin(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value


class EngineService:
    """Own one isolated search engine and three-way index set per part library."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engines: dict[str, Any] = {}
        self._fingerprints: dict[str, tuple[object, ...]] = {}
        self._lock = threading.RLock()
        self.operation_lock = threading.RLock()

    @property
    def _engine(self) -> Any | None:
        """Backward-compatible access to the default-library engine."""
        return self._engines.get("default")

    @staticmethod
    def capabilities() -> dict[str, bool]:
        try:
            import stp_similarity as core

            return {
                "has_faiss": bool(getattr(core, "HAS_FAISS", False)),
                "has_render": bool(getattr(core, "HAS_RENDER", False)),
            }
        except Exception:
            return {"has_faiss": False, "has_render": False}

    def initialized_count(self) -> int:
        with self._lock:
            return len(self._engines)

    def _index_path(self, library_id: str) -> Path:
        if library_id == "default":
            path = self.settings.db_path
        else:
            path = self.settings.db_path / "libraries" / library_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _view_path(self, library_id: str) -> Path:
        path = self.settings.render_dir / "libraries" / library_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_engine(
        self,
        library_id: str = "default",
        api_key: str | None = None,
        base_url: str | None = None,
        llm_model: str | None = None,
    ) -> Any:
        embedding_key = self.settings.api_key
        if not embedding_key:
            raise APIError(400, "缺少 API_KEY；请配置 .env 或在搜索请求中提供 api_key")
        embedding_url = self.settings.base_url
        llm_key = api_key or self.settings.llm_api_key or embedding_key
        llm_url = base_url if base_url is not None else self.settings.llm_base_url
        model = llm_model or self.settings.llm_model
        index_path = self._index_path(library_id)
        fingerprint = (
            embedding_key,
            embedding_url,
            llm_key,
            llm_url,
            model,
            self.settings.llm_timeout,
            self.settings.llm_max_retries,
            self.settings.llm_response_format,
            self.settings.llm_thinking,
            str(index_path),
        )
        with self._lock:
            if (
                library_id not in self._engines
                or fingerprint != self._fingerprints.get(library_id)
            ):
                from stp_similarity import STPSearchEngine

                view_path = self._view_path(library_id)
                self._engines[library_id] = STPSearchEngine(
                    api_key=embedding_key,
                    base_url=embedding_url or None,
                    llm_api_key=llm_key,
                    llm_base_url=llm_url or None,
                    embedding_model=self.settings.embedding_model,
                    llm_model=model,
                    db_path=str(index_path),
                    view_dir=str(view_path),
                    render_cache_dir=str(view_path),
                    enable_render=True,
                    llm_timeout=self.settings.llm_timeout,
                    llm_max_retries=self.settings.llm_max_retries,
                    llm_response_format=self.settings.llm_response_format,
                    llm_thinking=self.settings.llm_thinking,
                )
                self._fingerprints[library_id] = fingerprint
            return self._engines[library_id]

    def reset(self, library_id: str | None = None) -> None:
        with self._lock:
            if library_id is None:
                self._engines.clear()
                self._fingerprints.clear()
            else:
                self._engines.pop(library_id, None)
                self._fingerprints.pop(library_id, None)

    @staticmethod
    def _faiss_count(index: Any | None) -> int:
        try:
            return int(index.index.ntotal) if index and index.index else 0
        except Exception:
            return 0

    def index_counts(
        self,
        library_id: str = "default",
        *,
        initialize: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, int]:
        engine = self._engines.get(library_id)
        if engine is None and initialize:
            engine = self.get_engine(library_id, api_key, base_url, llm_model)
        if engine is None:
            return {"text": 0, "geometric": 0, "visual": 0}
        try:
            text_count = int(engine.index.collection.count())
        except Exception:
            text_count = 0
        return {
            "text": text_count,
            "geometric": self._faiss_count(getattr(engine.index, "geo_index", None)),
            "visual": self._faiss_count(getattr(engine.index, "visual_index", None)),
        }

    def index_count(self, library_id: str = "default") -> int:
        return self.index_counts(library_id)["text"]

    def build_index(
        self,
        library_id: str,
        directory: Path,
        view_dir: Path | None = None,
    ) -> dict[str, int]:
        files = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in STP_EXTENSIONS
        ]
        if not files:
            raise APIError(400, f"零件库中没有 STP/STEP 文件: {directory}")
        engine = self.get_engine(library_id)
        actual_view_dir = view_dir or self._view_path(library_id)
        with self.operation_lock:
            engine.build_index(str(directory), view_dir=str(actual_view_dir))
        return self.index_counts(library_id)

    def remove_part_from_index(
        self, library_id: str, filepath: str | Path
    ) -> tuple[dict[str, bool], dict[str, int]]:
        """不调用外部模型，直接从当前持久化三路索引删除一个零件。"""
        engine = self.get_engine(library_id)
        with self.operation_lock:
            removed = engine.index.delete_file(str(Path(filepath).resolve()))
            counts = self.index_counts(library_id)
        return removed, counts

    def search(
        self,
        library_id: str,
        query_path: Path,
        request: Any,
        reference_images: list[str],
        query_views: list[str] | None = None,
    ) -> list[dict]:
        engine = self.get_engine(
            library_id, request.api_key, request.base_url, request.llm_model
        )
        counts = self.index_counts(library_id)
        if counts["text"] <= 0:
            raise APIError(
                409,
                f"零件库 {library_id} 的索引为空，请先上传 library 文件并建立三路索引",
            )
        report_path = None
        if request.save_report:
            report_path = self.settings.report_dir / (
                f"{library_id}_{query_path.stem}_search_report.docx"
            )
        result_limit = request.effective_result_limit
        # Fetch one additional result so a query selected from the same library
        # can be removed without reducing the requested result count.
        engine_limit = min(50, result_limit + 1)
        with self.operation_lock:
            results = engine.search(
                query_path=str(query_path),
                coarse_top=max(request.coarse_top, engine_limit),
                final_top=engine_limit,
                use_llm_rerank=request.use_llm_rerank,
                use_vision=request.use_vision,
                use_geometric=request.use_geometric,
                use_feature_extraction=request.use_feature_extraction,
                use_three_way=request.use_three_way,
                use_3d_vision=request.use_3d_vision,
                auto_render=request.auto_render,
                save_report=request.save_report,
                report_path=str(report_path) if report_path else None,
                custom_query_text=request.custom_query_text,
                reference_images=reference_images or None,
                reference_dimensions=request.reference_dimensions,
                text_recall_k=request.text_recall_k,
                geo_recall_k=request.geo_recall_k,
                visual_recall_k=request.visual_recall_k,
                fusion_top_k=max(request.fusion_top_k, engine_limit),
                query_views=query_views,
            )
        return to_builtin(results)


class RenderService:
    VIEW_NAMES = ("front", "back", "top", "bottom", "left", "right")

    def __init__(self, settings: Settings):
        self.settings = settings
        self._renderers: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _cache_root(self, library_id: str | None = None) -> Path:
        if library_id:
            return self.settings.render_dir / "libraries" / library_id
        return self.settings.render_dir

    def _get_renderer(self, cache_dir: Path | None = None) -> Any:
        from stp_similarity import HAS_RENDER, STPRenderer

        if not HAS_RENDER:
            raise CapabilityUnavailableError(
                "在线渲染不可用：请安装 pyvista、imageio 和 pythonocc-core，并配置服务器离屏渲染环境"
            )
        cache_root = (cache_dir or self.settings.render_dir).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = str(cache_root)
        with self._lock:
            if cache_key not in self._renderers:
                self._renderers[cache_key] = STPRenderer(cache_dir=cache_key)
        return self._renderers[cache_key]

    def _media_url(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.settings.render_dir.resolve())
        except ValueError as exc:
            raise APIError(500, "渲染结果位于媒体目录之外") from exc
        encoded = "/".join(quote(part) for part in relative.parts)
        return f"/media/renders/{encoded}"

    def view_paths(self, stp_path: Path, library_id: str | None = None) -> list[str]:
        """Return canonical local paths in stable semantic order."""
        directory = self._cache_root(library_id) / stp_path.stem
        return [str(directory / f"{stp_path.stem}_{name}.png") for name in self.VIEW_NAMES]

    def render(
        self,
        stp_path: Path,
        force: bool,
        include_rotation: bool,
        rotation_frames: int,
        library_id: str | None = None,
    ) -> tuple[dict[str, str], str | None, bool]:
        cache_root = self._cache_root(library_id)
        renderer = self._get_renderer(cache_root)
        expected = cache_root / stp_path.stem
        from view_alignment import render_cache_valid
        cached = render_cache_valid(stp_path, expected)
        with self._lock:
            paths = renderer.render_to_cache(str(stp_path), force=force)
            if not paths:
                raise APIError(500, "STP 文件渲染失败，请检查文件有效性和 CAD 渲染依赖")
            rotation_path = None
            if include_rotation:
                rotation_path = renderer.render_rotation_gif(
                    str(stp_path), force=force, num_frames=rotation_frames
                )
        views: dict[str, str] = {}
        for path in paths:
            name = next(
                (view for view in self.VIEW_NAMES if Path(path).stem.endswith(f"_{view}")),
                Path(path).stem,
            )
            views[name] = self._media_url(path)
        return views, self._media_url(rotation_path) if rotation_path else None, cached and not force

    def match(self, query_path: Path, candidate_path: Path,
              query_library_id: str | None = None, candidate_library_id: str | None = None,
              method: str = "rigid24", force: bool = False) -> dict:
        from view_alignment import align_view_files, named_views

        with self._lock:
            query_renderer = self._get_renderer(self._cache_root(query_library_id))
            candidate_renderer = self._get_renderer(self._cache_root(candidate_library_id))
            query = query_renderer.render_to_cache(str(query_path), force=force)
            candidate = candidate_renderer.render_to_cache(str(candidate_path), force=force)
            if not query or not candidate:
                raise APIError(500, "六视图配对失败：查询件或候选件无法渲染")
            try:
                result = align_view_files(query, candidate, self.settings.render_dir / "alignments", method)
            except ValueError as exc:
                raise APIError(422, str(exc)) from exc
            # Do not mutate the persistent result (it retains local paths).
            return {**result,
                    "query_views": {n: self._media_url(p) for n, p in named_views(query).items()},
                    "candidate_views": {n: self._media_url(p) for n, p in named_views(candidate).items()},
                    "aligned_views": {n: self._media_url(p) for n, p in result["aligned_views"].items()}}

    def delete_cache(self, stp_path: Path, library_id: str | None = None) -> bool:
        """删除指定零件的六视图和旋转GIF缓存。"""
        import shutil

        cache_root = self._cache_root(library_id).resolve()
        target = (cache_root / stp_path.stem).resolve()
        try:
            target.relative_to(cache_root)
        except ValueError as exc:
            raise APIError(500, "渲染缓存路径越界，拒绝删除") from exc
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True


class LTRService:
    def __init__(self, settings: Settings, engines: EngineService):
        self.settings = settings
        self.engines = engines
        self._pipeline: Any | None = None
        self._model_path: Path | None = None
        self._lock = threading.RLock()

    @staticmethod
    def capability() -> bool:
        try:
            import stp_ltr  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def loaded(self) -> bool:
        return bool(self._pipeline and getattr(self._pipeline.model, "is_trained", False))

    @staticmethod
    def safe_model_name(name: str, model_type: str) -> str:
        safe = re.sub(r"[^\w.\-]", "_", Path(name).name)
        suffix = ".txt" if model_type == "lightgbm" else ".pkl"
        if not safe.lower().endswith(suffix):
            safe += suffix
        return safe

    def _info_cache(self) -> dict[str, Any]:
        engine = self.engines.get_engine()
        return engine.index._info_cache

    def create(self, model_type: str, model_path: Path | None = None) -> Any:
        if not self.capability():
            raise CapabilityUnavailableError("LTR 模块或其依赖未安装")
        from stp_ltr import LTRPipeline

        with self._lock:
            self._pipeline = LTRPipeline(
                model_type=model_type,
                model_path=str(model_path) if model_path else None,
                info_cache=self._info_cache(),
            )
            self._model_path = model_path
        return self._pipeline

    def require_pipeline(self) -> Any:
        if not self.loaded:
            raise APIError(409, "LTR 模型未训练或加载，请先调用 /api/ltr/train")
        return self._pipeline

    def train(self, request: Any, annotation: Path, evaluation: Path | None) -> tuple[str, dict]:
        model_name = self.safe_model_name(request.model_name, request.model_type)
        model_path = self.settings.model_dir / model_name
        pipeline = self.create(request.model_type)
        try:
            from stp_ltr import LightGBMModel

            if isinstance(pipeline.model, LightGBMModel):
                pipeline.model.params["num_leaves"] = request.num_leaves
                pipeline.model.params["learning_rate"] = request.learning_rate
                pipeline.model.num_iterations = request.num_iterations
        except ImportError:
            pass
        with self._lock:
            history = pipeline.train(
                annotation_path=str(annotation),
                eval_annotation_path=str(evaluation) if evaluation else None,
                query_infos=self._info_cache(),
            )
            pipeline.save_model(str(model_path))
            self._model_path = model_path
        return model_name, to_builtin(history)

    def evaluate(self, annotation: Path) -> dict[str, Any]:
        pipeline = self.require_pipeline()
        with self._lock:
            return to_builtin(
                pipeline.evaluate(str(annotation), query_infos=self._info_cache())
            )

    def feature_importance(self) -> dict[str, float]:
        pipeline = self.require_pipeline()
        return to_builtin(pipeline.feature_importance())

    def predict(self, query_path: Path, candidates: list[tuple[str, Path]]) -> list[dict]:
        pipeline = self.require_pipeline()
        from stp_similarity import parse_stp_deep

        query_info = parse_stp_deep(str(query_path))
        rows = [
            {
                "_file_id": file_id,
                "filepath": str(path),
                "path": str(path),
                "filename": path.name,
                "text_similarity": 0.5,
                "visual_similarity": 0.0,
                "recall_count": 1,
                "recall_sources": ["text"],
                "text_rank": index + 1,
                "geo_rank": 0,
                "visual_rank": 0,
            }
            for index, (file_id, path) in enumerate(candidates)
        ]
        with self._lock:
            reranked = pipeline.rerank(query_info, rows, self._info_cache())
        return [
            {
                "file_id": row["_file_id"],
                "filename": row["filename"],
                "ltr_score": float(row.get("ltr_score", 0.0)),
            }
            for row in reranked
        ]
