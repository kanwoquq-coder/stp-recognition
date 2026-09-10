"""Canonical orthographic CAD views and rotation-consistent image assignment.

No CAD/OpenGL import at module scope: matching also works on CPU-only servers.
Coordinates are right handed; normals point from the object towards the camera.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

RENDER_VERSION = "canonical-ortho-v2"
VIEW_AXES = {
    "front": ((0, 0, 1), (0, 1, 0)),
    "back": ((0, 0, -1), (0, 1, 0)),
    "top": ((0, 1, 0), (0, 0, -1)),
    "bottom": ((0, -1, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 1, 0)),
    "right": ((1, 0, 0), (0, 1, 0)),
}
VIEW_NAMES = tuple(VIEW_AXES)


def canonical_frame(points, triangles):
    """Area-weighted surface frame, preferring CAD plane normals over PCA.

Large planar faces define Z; the largest perpendicular plane defines X.
This avoids tessellation-density bias and PCA diagonal axes on U/L brackets.
Surface moments provide a fallback for curved shapes. Axis sign ambiguities
are intentional and resolved by the subsequent 24-rotation search.
"""
    points = np.asarray(points, dtype=float)
    tri = points[np.asarray(triangles, dtype=int)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    weights = np.linalg.norm(cross, axis=1)
    valid = weights > np.finfo(float).eps
    if not valid.any() or not np.isfinite(points).all():
        raise ValueError("网格为空、退化或含非有限坐标，无法建立坐标系")
    tri, cross, weights = tri[valid], cross[valid], weights[valid]
    normals = cross / weights[:, None]
    weights /= weights.sum()
    center = np.einsum("i,ij->j", weights, tri.mean(axis=1))
    # Exact second moment of a uniformly sampled triangle surface.
    shifted = tri - center
    sums = shifted.sum(axis=1)
    moment = (np.einsum("n,nvi,nvj->ij", weights, shifted, shifted)
              + np.einsum("n,ni,nj->ij", weights, sums, sums)) / 12.0
    eigenvalues, eigenvectors = np.linalg.eigh(moment)

    # Cluster unoriented normals within 3 degrees, area ordered, bounded cost.
    clusters = []
    for i in np.argsort(-weights, kind="stable"):
        normal = normals[i]
        if clusters and max(abs(np.dot(normal, n)) for n in clusters) > 0.99863:
            continue
        clusters.append(normal)
        if len(clusters) >= 64:
            break
    support = [float(weights[np.abs(normals @ n) > 0.99863].sum()) for n in clusters]
    z = clusters[int(np.argmax(support))].copy()
    method = "dominant_surface_normals"
    if max(support) < 0.15:
        z = eigenvectors[:, 0].copy()
        method = "surface_pca"
    perpendicular = [(s, n) for s, n in zip(support, clusters) if abs(n @ z) < 0.05]
    if perpendicular and max(s for s, _ in perpendicular) >= 0.05:
        x = max(perpendicular, key=lambda item: item[0])[1].copy()
    else:
        projection = np.eye(3) - np.outer(z, z)
        _, axes = np.linalg.eigh(projection @ moment @ projection)
        x = axes[:, -1]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    basis = np.column_stack((x, y, z))
    local = (points - center) @ basis
    # Largest extent -> Y, middle -> X, smallest -> Z: front normally shows
    # the broad side. Remaining ties are handled by the rigid pairing stage.
    spans = np.ptp(local, axis=0)
    order = np.argsort(spans, kind="stable")
    basis = basis[:, [order[1], order[2], order[0]]]
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    local = (points - center) @ basis
    # Resolve non-symmetric signs using area-weighted third moments.
    local_centers = (tri.mean(axis=1) - center) @ basis
    for axis in (0, 2):
        third = np.sum(weights * local_centers[:, axis] ** 3)
        if third < -1e-8 * max(np.ptp(local[:, axis]), 1e-12) ** 3:
            basis[:, axis] *= -1
    basis[:, 1] = np.cross(basis[:, 2], basis[:, 0])
    local = (points - center) @ basis
    bbox_center = (local.min(axis=0) + local.max(axis=0)) / 2
    origin = center + basis @ bbox_center
    local = (points - origin) @ basis
    dims = np.ptp(local, axis=0)
    return local, {
        "version": RENDER_VERSION, "method": method,
        "origin": origin.tolist(), "basis_columns": basis.tolist(),
        "dimensions": dims.tolist(), "unit": "source_mesh_unit",
        "pca_eigenvalues": eigenvalues.tolist(),
        "axis_ambiguity_possible": True,
        "projection": "orthographic", "image_size": [512, 512],
    }


def source_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def render_cache_valid(source, directory):
    directory = Path(directory)
    try:
        meta = json.loads((directory / "render_manifest.json").read_text(encoding="utf-8"))
        return (meta["version"] == RENDER_VERSION
                and meta["source"] == source_signature(source)
                and all((directory / f"{Path(source).stem}_{n}.png").is_file() for n in VIEW_NAMES))
    except (OSError, ValueError, KeyError):
        return False


def render_standard_views(mesh, output_dir, base_name):
    import pyvista as pv

    # Some PyVista filters return the input object for already-triangular data.
    mesh = mesh.copy(deep=True)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    local, metadata = canonical_frame(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    mesh.points = local
    scale = float(np.max(np.ptp(local, axis=0))) * 0.60
    if scale <= 0:
        raise ValueError("零件尺寸为零")
    metadata["parallel_scale"] = scale
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    # Use one render window per direction.  With an off-screen VTK window,
    # changing the camera after the first screenshot can leave the previous
    # frame in the render buffer on some Mesa/EGL builds.  The result looks
    # like six correctly named PNGs, but every file contains the first view.
    # A fresh plotter plus explicit VTK camera setters makes every screenshot
    # independent and works consistently on Windows and headless Linux.
    for name, (normal, up) in VIEW_AXES.items():
        plotter = pv.Plotter(off_screen=True, window_size=[512, 512])
        try:
            plotter.add_mesh(mesh, color="#A7C1D1", show_edges=False,
                             smooth_shading=False, specular=0.0, ambient=0.45)
            plotter.set_background("white")
            camera = plotter.camera
            position = np.asarray(normal, dtype=float) * scale * 5.0
            camera.SetPosition(*position)
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetViewUp(*up)
            camera.SetParallelProjection(True)
            camera.SetParallelScale(scale)
            plotter.reset_camera_clipping_range()
            plotter.render()
            path = directory / f"{base_name}_{name}.png"
            plotter.screenshot(str(path))
            paths.append(str(path))
        finally:
            plotter.close()
    return paths, metadata


def named_views(paths):
    if isinstance(paths, dict):
        return {n: str(paths[n]) for n in VIEW_NAMES if paths.get(n)}
    found = {}
    for path in paths or []:
        if not path:
            continue
        stem = Path(path).stem.lower()
        aliases = {"front": "主视图", "back": "后视图", "top": "俯视图",
                   "bottom": "仰视图", "left": "左视图", "right": "右视图"}
        for name in VIEW_NAMES:
            if any(stem == value or stem.endswith("_" + value) for value in (name, aliases[name])):
                found[name] = str(path)
    return found


def cube_rotations():
    """The 24 signed axis permutations with determinant +1; no reflections."""
    result = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.eye(3, dtype=int)[:, perm] * signs
            if round(np.linalg.det(rotation)) == 1:
                result.append(rotation)
    # Identity first gives deterministic natural order for symmetric parts.
    return sorted(result, key=lambda r: (not np.array_equal(r, np.eye(3)), tuple(r.flat)))


def rotation_assignment(rotation):
    """Map query view -> (candidate view, counterclockwise image quarter turns)."""
    result = []
    for qname, (qn, qu) in VIEW_AXES.items():
        qn, qu = np.array(qn), np.array(qu)
        qr = np.cross(qu, qn)
        for cname, (cn, cu) in VIEW_AXES.items():
            if not np.array_equal(rotation @ cn, qn):
                continue
            transformed_right = rotation @ np.cross(cu, cn)
            angle = np.arctan2(transformed_right @ qu, transformed_right @ qr)
            result.append((qname, cname, int(round(angle / (np.pi / 2))) % 4))
            break
    return result


def _read_feature(path):
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "white")
        bg.alpha_composite(rgba)
        rgb = np.asarray(bg.convert("RGB"))
    mask = np.min(rgb, axis=2) < 240
    if mask.sum() < 8:
        raise ValueError(f"视图为空白或前景太小: {Path(path).name}")
    yy, xx = np.where(mask)
    # Shared scale is retained; only integer translation centers the foreground.
    side = max(rgb.shape[:2])
    image = Image.fromarray(rgb)
    canvas = Image.new("RGB", (side, side), "white")
    dx = int(round((side - 1 - xx.min() - xx.max()) / 2))
    dy = int(round((side - 1 - yy.min() - yy.max()) / 2))
    canvas.paste(image, (dx, dy))
    arr = np.asarray(canvas.resize((96, 96), Image.Resampling.LANCZOS), dtype=float)
    return np.min(arr, axis=2) < 240, 1.0 - arr.mean(axis=2) / 255.0


def _similarity(a, b, turns):
    am, ag = a
    bm, bg = (np.rot90(v, turns) for v in b)
    union = np.logical_or(am, bm)
    iou = np.logical_and(am, bm).sum() / max(union.sum(), 1)
    # Silhouette includes through-holes. Shading is a weak secondary cue only.
    gray = max(0.0, 1.0 - float(np.abs(ag[union] - bg[union]).mean()) * 3)
    return float(0.85 * iou + 0.15 * gray)


def match_views(query_paths, candidate_paths, method="rigid24"):
    if method not in {"rigid24", "hungarian"}:
        raise ValueError("配对方式必须为 rigid24 或 hungarian")
    query, candidate = named_views(query_paths), named_views(candidate_paths)
    if set(query) != set(VIEW_NAMES) or set(candidate) != set(VIEW_NAMES):
        raise ValueError("配对需要查询件和候选件各有完整、具名的六视图")
    qfeat = [_read_feature(query[n]) for n in VIEW_NAMES]
    cfeat = [_read_feature(candidate[n]) for n in VIEW_NAMES]
    scores = np.array([[[_similarity(a, b, k) for k in range(4)] for b in cfeat] for a in qfeat])
    row, col = linear_sum_assignment(scores.max(axis=2), maximize=True)
    upper = float(scores.max(axis=2)[row, col].mean())
    rotations = []
    for rotation in cube_rotations():
        assignment = rotation_assignment(rotation)
        score = np.mean([scores[VIEW_NAMES.index(q), VIEW_NAMES.index(c), k] for q, c, k in assignment])
        rotations.append((float(score), rotation, assignment))
    rotations.sort(key=lambda item: item[0], reverse=True)
    score, rotation, assignment = rotations[0]
    gap = score - rotations[1][0]
    if method == "hungarian":
        assignment = [(VIEW_NAMES[i], VIEW_NAMES[j], int(scores[i, j].argmax())) for i, j in zip(row, col)]
        score, rotation = upper, None
        gap = None
    pairs = [{"query_view": q, "candidate_view": c, "rotation_degrees_ccw": 90 * k,
              "score": round(float(scores[VIEW_NAMES.index(q), VIEW_NAMES.index(c), k]), 6)}
             for q, c, k in assignment]
    return {"status": "ready", "method": method, "score": round(score, 6),
            "hungarian_score": round(upper, 6),
            "rotation_matrix": rotation.tolist() if rotation is not None else None,
            "rigid_consistent": method == "rigid24", "score_gap": gap,
            "ambiguous": gap is None or gap < 0.015, "pairs": pairs,
            "warning": ("配对不是语义前视图或真实相似概率；对称件可能有多个等价方向。"
                        if method == "rigid24" else "独立配对不保证六视图来自同一三维姿态。")}


def align_view_files(query_paths, candidate_paths, output_root, method="rigid24"):
    """Persist aligned copies; never overwrite an original render or CAD file."""
    query, candidate = named_views(query_paths), named_views(candidate_paths)
    signatures = {"version": RENDER_VERSION, "method": method,
                  "query": {n: source_signature(p) for n, p in query.items()},
                  "candidate": {n: source_signature(p) for n, p in candidate.items()}}
    key = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(output_root) / key
    metadata_file = directory / "alignment.json"
    try:
        result = json.loads(metadata_file.read_text(encoding="utf-8"))
        if all(Path(result["aligned_views"][n]).is_file() for n in VIEW_NAMES):
            return result
    except (OSError, KeyError, ValueError):
        pass
    result = match_views(query, candidate, method)
    directory.mkdir(parents=True, exist_ok=True)
    aligned = {}
    for pair in result["pairs"]:
        path = directory / ("aligned_" + pair["query_view"] + ".png")
        with Image.open(candidate[pair["candidate_view"]]) as im:
            turns = pair["rotation_degrees_ccw"] // 90
            if turns:
                im = im.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                                   3: Image.Transpose.ROTATE_270}[turns])
            im.save(path)
        aligned[pair["query_view"]] = str(path.resolve())
    result["aligned_views"] = aligned
    metadata_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
