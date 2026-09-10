from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.config import Settings
from app.errors import APIError, FileReferenceError
from app.schemas import (
    FileCategory,
    FileItem,
    IndexSummary,
    LibraryItem,
    PartListResponse,
)


_ALLOWED_EXTENSIONS = {
    "query": {".stp", ".step"},
    "library": {".stp", ".step"},
    "reference": {".png", ".jpg", ".jpeg", ".webp"},
    "annotation": {".json"},
}
DEFAULT_LIBRARY_ID = "default"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_index() -> dict[str, Any]:
    return {
        "status": "not_built",
        "text_count": 0,
        "geometric_count": 0,
        "visual_count": 0,
        "indexed_at": None,
        "error": None,
    }


class FileStore:
    """Persistent file catalog and logical part-library registry."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.metadata_path = settings.runtime_dir / "files.json"
        self.libraries_path = settings.runtime_dir / "libraries.json"
        self._lock = threading.RLock()
        self._libraries: dict[str, dict[str, Any]] = self._load_json(
            self.libraries_path
        )
        self._ensure_default_library()
        self._records: dict[str, dict[str, Any]] = self._load_json(
            self.metadata_path
        )
        self._migrate_records()

    @staticmethod
    def _load_json(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _persist_json(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_path.replace(path)

    def _persist(self) -> None:
        self._persist_json(self.metadata_path, self._records)

    def _persist_libraries(self) -> None:
        self._persist_json(self.libraries_path, self._libraries)

    def _ensure_default_library(self) -> None:
        now = _utcnow()
        default = self._libraries.setdefault(
            DEFAULT_LIBRARY_ID,
            {
                "library_id": DEFAULT_LIBRARY_ID,
                "name": "默认零件库",
                "description": "兼容旧接口的默认零件库",
                "created_at": now,
                "updated_at": now,
                "index": _empty_index(),
            },
        )
        default.setdefault("index", _empty_index())
        self._library_dir(DEFAULT_LIBRARY_ID).mkdir(parents=True, exist_ok=True)
        self._persist_libraries()

    def _migrate_records(self) -> None:
        """Attach old library uploads to the default library without losing data."""
        changed = False
        default_dir = self._library_dir(DEFAULT_LIBRARY_ID)
        for record in self._records.values():
            record.setdefault("metadata", None)
            record.setdefault("index_status", "not_built")
            record.setdefault("indexed_at", None)
            if record.get("category") != "library":
                record.setdefault("library_id", None)
                continue
            if not record.get("library_id"):
                record["library_id"] = DEFAULT_LIBRARY_ID
                changed = True
            source = Path(record.get("path", ""))
            if (
                record["library_id"] == DEFAULT_LIBRARY_ID
                and source.is_file()
                and source.parent.resolve() == self.settings.library_dir.resolve()
            ):
                target = default_dir / source.name
                try:
                    if not target.exists():
                        source.replace(target)
                    record["path"] = str(target.resolve())
                    changed = True
                except OSError:
                    # The record remains usable even if an old file cannot be moved.
                    pass
        if changed:
            self._persist()

    @staticmethod
    def _safe_name(filename: str) -> str:
        name = Path(filename or "upload").name
        name = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", name)
        return name[:180] or "upload"

    @staticmethod
    def _safe_library_id(library_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", library_id):
            raise APIError(422, "library_id 格式无效")
        return library_id

    def _library_dir(self, library_id: str) -> Path:
        return self.settings.library_dir / self._safe_library_id(library_id)

    def _category_dir(
        self, category: FileCategory, library_id: str | None = None
    ) -> Path:
        if category == "library":
            return self.library_directory(library_id or DEFAULT_LIBRARY_ID)
        path = self.settings.upload_dir / f"{category}s"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_library(self, name: str, description: str = "") -> LibraryItem:
        clean_name = name.strip()
        if not clean_name:
            raise APIError(422, "零件库名称不能为空")
        with self._lock:
            if any(
                item.get("name", "").casefold() == clean_name.casefold()
                for item in self._libraries.values()
            ):
                raise APIError(409, f"零件库名称已存在: {clean_name}")
            library_id = uuid.uuid4().hex[:12]
            now = _utcnow()
            self._libraries[library_id] = {
                "library_id": library_id,
                "name": clean_name,
                "description": description.strip(),
                "created_at": now,
                "updated_at": now,
                "index": _empty_index(),
            }
            self._library_dir(library_id).mkdir(parents=True, exist_ok=True)
            self._persist_libraries()
        return self.library_item(library_id)

    def library(self, library_id: str) -> dict[str, Any]:
        library_id = self._safe_library_id(library_id)
        with self._lock:
            record = self._libraries.get(library_id)
            if not record:
                raise FileReferenceError(f"零件库不存在: {library_id}")
            return dict(record)

    def library_directory(self, library_id: str) -> Path:
        self.library(library_id)
        path = self._library_dir(library_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def library_item(self, library_id: str) -> LibraryItem:
        library = self.library(library_id)
        with self._lock:
            parts = [
                item
                for item in self._records.values()
                if item.get("category") == "library"
                and item.get("library_id") == library_id
            ]
        payload = {key: value for key, value in library.items() if key != "index"}
        return LibraryItem(
            **payload,
            part_count=len(parts),
            indexed_part_count=sum(
                item.get("index_status") == "ready" for item in parts
            ),
            index=IndexSummary.model_validate(library.get("index") or _empty_index()),
        )

    def list_libraries(self) -> list[LibraryItem]:
        with self._lock:
            ids = list(self._libraries)
        items = [self.library_item(library_id) for library_id in ids]
        items.sort(key=lambda item: (item.library_id != DEFAULT_LIBRARY_ID, item.name))
        return items

    def index_summary(self, library_id: str) -> IndexSummary:
        return self.library_item(library_id).index

    def set_library_index_state(
        self,
        library_id: str,
        status: str,
        *,
        text_count: int = 0,
        geometric_count: int = 0,
        visual_count: int = 0,
        error: str | None = None,
    ) -> IndexSummary:
        if status not in {"not_built", "building", "ready", "failed"}:
            raise ValueError(f"unsupported index status: {status}")
        now = _utcnow()
        with self._lock:
            library = self._libraries.get(library_id)
            if not library:
                raise FileReferenceError(f"零件库不存在: {library_id}")
            previous = library.get("index") or _empty_index()
            index = {
                "status": status,
                "text_count": text_count if status == "ready" else previous.get("text_count", 0),
                "geometric_count": (
                    geometric_count if status == "ready" else previous.get("geometric_count", 0)
                ),
                "visual_count": (
                    visual_count if status == "ready" else previous.get("visual_count", 0)
                ),
                "indexed_at": now if status == "ready" else previous.get("indexed_at"),
                "error": error,
            }
            library["index"] = index
            library["updated_at"] = now
            for record in self._records.values():
                if (
                    record.get("category") == "library"
                    and record.get("library_id") == library_id
                ):
                    record["index_status"] = status
                    if status == "ready":
                        record["indexed_at"] = now
            self._persist_libraries()
            self._persist()
        return IndexSummary.model_validate(index)

    async def save(
        self,
        upload: UploadFile,
        category: FileCategory,
        library_id: str | None = None,
    ) -> FileItem:
        original_name = self._safe_name(upload.filename or "upload")
        extension = Path(original_name).suffix.lower()
        if extension not in _ALLOWED_EXTENSIONS[category]:
            allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS[category]))
            raise APIError(415, f"{category} 类型仅允许上传: {allowed}")

        assigned_library = None
        if category == "library":
            assigned_library = library_id or DEFAULT_LIBRARY_ID
            self.library(assigned_library)
        elif library_id:
            raise APIError(422, "只有 category=library 时才能指定 library_id")

        file_id = uuid.uuid4().hex
        stored_path = self._category_dir(category, assigned_library) / (
            f"{file_id}_{original_name}"
        )
        size = 0
        try:
            with stored_path.open("wb") as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise APIError(413, "上传文件超过大小限制")
                    target.write(chunk)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        record = {
            "file_id": file_id,
            "original_name": original_name,
            "category": category,
            "size": size,
            "content_type": upload.content_type or "application/octet-stream",
            "created_at": _utcnow(),
            "path": str(stored_path.resolve()),
            "library_id": assigned_library,
            "index_status": "not_built",
            "indexed_at": None,
            "metadata": None,
        }
        with self._lock:
            self._records[file_id] = record
            if assigned_library:
                self._libraries[assigned_library]["updated_at"] = _utcnow()
                self._libraries[assigned_library]["index"]["status"] = "not_built"
                self._libraries[assigned_library]["index"]["error"] = None
                self._persist_libraries()
            self._persist()
        return self.public_item(record)

    def update_part_metadata(
        self, file_id: str, metadata: dict[str, Any]
    ) -> FileItem:
        with self._lock:
            record = self._records.get(file_id)
            if not record:
                raise FileReferenceError(f"文件不存在: {file_id}")
            record["metadata"] = metadata
            self._persist()
            return self.public_item(record)

    def delete(self, file_id: str) -> tuple[dict[str, Any], bool]:
        """删除磁盘文件及文件目录记录，返回删除前记录和文件删除状态。"""
        with self._lock:
            record = self._records.get(file_id)
            if not record:
                raise FileReferenceError(f"文件不存在: {file_id}")
            snapshot = dict(record)
            stored_path = Path(record["path"])
            try:
                existed = stored_path.is_file()
                stored_path.unlink(missing_ok=True)
            except OSError as exc:
                raise APIError(500, f"删除零件原文件失败: {exc}") from exc

            self._records.pop(file_id, None)
            library_id = snapshot.get("library_id")
            if library_id and library_id in self._libraries:
                self._libraries[library_id]["updated_at"] = _utcnow()
                self._persist_libraries()
            self._persist()
            return snapshot, existed

    def clear_library_index_state(self, library_id: str) -> IndexSummary:
        """将空库或失效库的索引状态和计数全部归零。"""
        with self._lock:
            library = self._libraries.get(library_id)
            if not library:
                raise FileReferenceError(f"零件库不存在: {library_id}")
            library["index"] = _empty_index()
            library["updated_at"] = _utcnow()
            for record in self._records.values():
                if record.get("library_id") == library_id:
                    record["index_status"] = "not_built"
                    record["indexed_at"] = None
            self._persist_libraries()
            self._persist()
            return IndexSummary.model_validate(library["index"])

    def public_item(self, record: dict[str, Any]) -> FileItem:
        library_id = record.get("library_id")
        library_name = None
        if library_id:
            library_name = self._libraries.get(library_id, {}).get("name")
        return FileItem(
            file_id=record["file_id"],
            original_name=record["original_name"],
            category=record["category"],
            size=record["size"],
            content_type=record["content_type"],
            created_at=record["created_at"],
            download_url=f"/api/files/{record['file_id']}/download",
            library_id=library_id,
            library_name=library_name,
            index_status=record.get("index_status", "not_built"),
            indexed_at=record.get("indexed_at"),
            metadata=record.get("metadata"),
        )

    def list(
        self,
        category: FileCategory | None = None,
        library_id: str | None = None,
    ) -> list[FileItem]:
        if library_id:
            self.library(library_id)
        with self._lock:
            records = list(self._records.values())
        if category:
            records = [record for record in records if record["category"] == category]
        if library_id:
            records = [
                record for record in records if record.get("library_id") == library_id
            ]
        records.sort(key=lambda item: item["created_at"], reverse=True)
        return [self.public_item(record) for record in records]

    def list_parts(
        self,
        library_id: str | None,
        keyword: str | None,
        offset: int,
        limit: int,
    ) -> PartListResponse:
        records = self.list(category="library", library_id=library_id)
        if keyword:
            needle = keyword.casefold()
            records = [
                record
                for record in records
                if needle in record.original_name.casefold()
            ]
        total = len(records)
        return PartListResponse(
            library_id=library_id,
            total=total,
            offset=offset,
            limit=limit,
            items=records[offset : offset + limit],
        )

    def record(self, file_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(file_id)
        if not record:
            raise FileReferenceError(f"文件不存在: {file_id}")
        path = Path(record["path"])
        if not path.is_file():
            raise FileReferenceError(f"文件记录存在但磁盘文件缺失: {file_id}")
        return record

    def file_id_for_path(self, path: str | Path) -> str | None:
        candidate = Path(path)
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            resolved = candidate
        with self._lock:
            records = list(self._records.values())
        for record in records:
            try:
                if Path(record["path"]).resolve() == resolved:
                    return record["file_id"]
            except (OSError, ValueError):
                continue
        prefix = candidate.name.split("_", 1)[0]
        return prefix if prefix in self._records else None

    def resolve_file(
        self,
        file_id: str | None,
        file_path: str | None,
        extensions: set[str] | None = None,
    ) -> Path:
        if file_id:
            path = Path(self.record(file_id)["path"])
        elif file_path and self.settings.allow_local_paths:
            path = Path(file_path).expanduser().resolve()
            if not path.is_file():
                raise FileReferenceError(f"本地文件不存在: {path}")
        elif file_path:
            raise APIError(403, "本地路径模式已关闭，请先上传文件并使用 file_id")
        else:
            raise APIError(422, "必须提供 file_id 或 file_path")
        if extensions and path.suffix.lower() not in extensions:
            raise APIError(415, f"不支持的文件类型: {path.suffix}")
        return path

    def resolve_annotation(self, file_id: str | None, file_path: str | None) -> Path:
        return self.resolve_file(file_id, file_path, {".json"})

    def resolve_index_directory(
        self, directory: str | None, library_id: str = DEFAULT_LIBRARY_ID
    ) -> Path:
        self.library(library_id)
        if not directory:
            return self.library_directory(library_id)
        if not self.settings.allow_local_paths:
            raise APIError(403, "本地路径模式已关闭；请上传 library 文件后构建零件库")
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            raise FileReferenceError(f"索引目录不存在: {path}")
        return path
