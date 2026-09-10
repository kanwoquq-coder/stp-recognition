"""Refresh persisted part metadata with the current feature recognizer.

This is an operator CLI, not an HTTP endpoint.  It keeps the existing API
contract unchanged while allowing records created by recognizer 1.0 to be
recomputed by recognizer 1.1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.storage import FileStore
from stp_similarity import parse_stp_deep


def _builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    return value


def _records(store: FileStore, file_id: str | None, library_id: str | None, all_files: bool):
    if file_id:
        return [store.record(file_id)]
    if library_id:
        return [store.record(item.file_id) for item in store.list("library", library_id)]
    if all_files:
        return [
            store.record(item.file_id)
            for category in ("library", "query")
            for item in store.list(category)
        ]
    raise ValueError("必须指定 --file-id、--library-id 或 --all")


def main() -> int:
    parser = argparse.ArgumentParser(description="使用特征识别1.1刷新已保存的零件元数据")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-id", help="只刷新一个文件ID")
    group.add_argument("--library-id", help="刷新指定零件库的全部零件")
    group.add_argument("--all", action="store_true", help="刷新全部零件库和查询件")
    args = parser.parse_args()

    store = FileStore(Settings.load())
    records = _records(store, args.file_id, args.library_id, args.all)
    succeeded = failed = 0
    for index, record in enumerate(records, 1):
        file_id = record["file_id"]
        path = Path(record["path"])
        try:
            metadata = _builtin(parse_stp_deep(str(path)))
            store.update_part_metadata(file_id, metadata)
            succeeded += 1
            mfg = metadata.get("mfg_features", {})
            print(
                f"[{index}/{len(records)}] OK {record['original_name']} "
                f"通孔={mfg.get('through_holes', 0)} "
                f"盲孔={mfg.get('blind_holes', 0)} "
                f"槽={mfg.get('slots', 0)} 型腔={mfg.get('pockets', 0)}"
            )
        except Exception as exc:
            failed += 1
            store.update_part_metadata(
                file_id, {"parse_error": f"{type(exc).__name__}: {exc}"}
            )
            print(
                f"[{index}/{len(records)}] FAILED {record['original_name']}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(f"刷新完成：成功 {succeeded}，失败 {failed}，总计 {len(records)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
