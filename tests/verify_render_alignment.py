"""Offline real-render verification. Run with the CAD/render Python environment.

Usage: python tests/verify_render_alignment.py [--step sample.stp] [--output path]
Creates only new verification artifacts; never edits the source STEP.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from view_alignment import render_standard_views, align_view_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step")
    parser.add_argument("--mesh", help="已转换的STL网格；可用于CAD和渲染依赖分离的环境")
    parser.add_argument("--export-only", action="store_true", help="仅把STEP转换为验证网格")
    parser.add_argument("--output", default="runtime/alignment_verification")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.mesh:
        mesh = pv.read(args.mesh)
    elif args.step:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        reader = STEPControl_Reader()
        assert reader.ReadFile(args.step) == 1
        reader.TransferRoots()
        shape = reader.OneShape()
        mesher = BRepMesh_IncrementalMesh(shape, 0.1)
        mesher.Perform()
        with tempfile.TemporaryDirectory() as temporary:
            stl = str(Path(temporary) / "part.stl")
            StlAPI_Writer().Write(shape, stl)
            mesh = pv.read(stl)
    else:
        mesh = pv.Box(bounds=(-3, 3, -4, 4, -0.4, 0.4)).merge(
            pv.Box(bounds=(-3, -2.5, -4, 4, 0.4, 2)))
    if args.export_only:
        output.mkdir(parents=True, exist_ok=True)
        mesh.save(output / "verification_input.stl")
        print(output / "verification_input.stl")
        return
    input_points = mesh.points.copy()
    original, meta = render_standard_views(mesh, output / "query", "query")
    np.testing.assert_array_equal(mesh.points, input_points)
    rotated = mesh.copy(deep=True)
    rotation = Rotation.from_euler("xyz", [33, -47, 71], degrees=True).as_matrix()
    rotated.points = np.asarray(mesh.points, dtype=float) @ rotation.T + [170, -280, 93]
    candidate, rotated_meta = render_standard_views(rotated, output / "candidate", "candidate")
    result = align_view_files(original, candidate, output / "aligned")
    np.testing.assert_allclose(meta["dimensions"], rotated_meta["dimensions"], atol=1e-3)
    assert result["score"] > 0.97, result
    (output / "verification.json").write_text(json.dumps({
        "input": args.step or args.mesh or "synthetic_bracket", "query_frame": meta,
        "candidate_frame": rotated_meta, "alignment": result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: arbitrary rotation + translation, six-view score={result['score']:.6f}")
    print(output)


if __name__ == "__main__":
    main()
