"""Batch renderer adapted from the local step.py implementation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from app.config import Settings
from stp_similarity import HAS_RENDER, STPRenderer


def batch_render(input_dir: Path, output_dir: Path, force: bool = False) -> int:
    if not HAS_RENDER:
        raise RuntimeError(
            "渲染依赖未安装：需要 pyvista、imageio 和 pythonocc-core"
        )
    renderer = STPRenderer(cache_dir=str(output_dir))
    files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".stp", ".step"}
    )
    if len({path.stem for path in files}) != len(files):
        raise ValueError("目录存在同名STP/STEP零件，缓存会冲突；请按零件库分别渲染或先通过上传接口生成唯一文件名")
    failures = []
    for path in files:
        result = renderer.render_to_cache(str(path), force=force)
        state = "OK" if result else "FAILED"
        print(f"[{state}] {path.name}")
        if not result:
            failures.append(path.name)
    if failures:
        raise RuntimeError(f"{len(failures)} 个零件渲染失败: {', '.join(failures[:10])}")
    return len(files)


def main() -> None:
    settings = Settings.load()
    parser = argparse.ArgumentParser(description="批量渲染 STP/STEP 六视图")
    parser.add_argument("input_dir", nargs="?", default=str(settings.library_dir))
    parser.add_argument("--output", default=str(settings.render_dir))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--library-id", help="升级指定零件库，自动使用该库STP与渲染目录；不重建向量索引")
    args = parser.parse_args()
    if args.library_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.library_id):
            parser.error("library-id 格式无效")
        args.input_dir = str(settings.library_dir / args.library_id)
        args.output = str(settings.render_dir / "libraries" / args.library_id)
    if not Path(args.input_dir).is_dir():
        parser.error("零件目录不存在")
    count = batch_render(Path(args.input_dir).resolve(), Path(args.output).resolve(), args.force)
    print(f"完成，共处理 {count} 个文件")


if __name__ == "__main__":
    main()
