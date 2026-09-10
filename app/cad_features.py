"""Deterministic STEP B-Rep feature recognition (no trained model required).

Version 1.1 reads the topology already present in a STEP Part 21 file.  It is
deliberately conservative: a cylindrical face is not a hole merely because it
is cylindrical.  Hole walls must be inward-facing or be connected to an inner
boundary of an exterior face, and coaxial wall segments are grouped as one
physical feature.

This module is internal.  Its result is mapped back to the existing API fields
by :mod:`stp_similarity`, so it does not change the HTTP contract.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


LEGACY_FEATURE_RECOGNITION_VERSION = "1.0"
FEATURE_RECOGNITION_VERSION = "1.1"

_SURFACE_TYPES = {
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
}
_BOUND_TYPES = {"FACE_BOUND", "FACE_OUTER_BOUND"}


@dataclass
class _Face:
    face_id: str
    surface_id: str
    surface_type: str
    same_sense: bool | None
    bounds: list[tuple[str, set[str]]]

    @property
    def edges(self) -> set[str]:
        result: set[str] = set()
        for _, edge_ids in self.bounds:
            result.update(edge_ids)
        return result


@dataclass
class _Cylinder:
    face_id: str
    axis_point: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    radius: float
    edge_ids: set[str]
    circle_edges: set[str]
    axial_interval: tuple[float, float] | None
    inward_facing: bool


def _refs(args: str) -> list[str]:
    return re.findall(r"#\d+", args or "")


def _numbers(args: str) -> list[float]:
    # Entity references are removed so '#123' cannot be mistaken for geometry.
    cleaned = re.sub(r"#\d+", "", args or "")
    result: list[float] = []
    for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", cleaned):
        try:
            result.append(float(token))
        except ValueError:
            continue
    return result


def _vector(args: str) -> tuple[float, float, float] | None:
    values = _numbers(args)
    if len(values) < 3:
        return None
    return float(values[-3]), float(values[-2]), float(values[-1])


def _add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _mul(a, scale: float):
    return a[0] * scale, a[1] * scale, a[2] * scale


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a) -> float:
    return math.sqrt(max(_dot(a, a), 0.0))


def _normalize(a) -> tuple[float, float, float] | None:
    length = _norm(a)
    if length <= 1e-12:
        return None
    value = _mul(a, 1.0 / length)
    # A line has no preferred direction.  Canonicalizing its sign makes
    # parallel axes emitted in opposite directions compare consistently.
    for component in value:
        if abs(component) <= 1e-10:
            continue
        return _mul(value, -1.0) if component < 0 else value
    return value


def _entity_type(entities: dict, ref: str) -> str:
    return (entities.get(ref) or {}).get("type", "")


def _resolve_edge_curves(
    ref: str, entities: dict, seen: set[str] | None = None
) -> set[str]:
    """Resolve FACE_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE."""
    if seen is None:
        seen = set()
    if ref in seen:
        return set()
    seen.add(ref)
    entity = entities.get(ref)
    if not entity:
        return set()
    if entity.get("type") == "EDGE_CURVE":
        return {ref}
    result: set[str] = set()
    for child in _refs(entity.get("args", "")):
        result.update(_resolve_edge_curves(child, entities, seen))
    return result


def _face_same_sense(args: str) -> bool | None:
    match = re.search(r"\.([TF])\.\s*$", args or "", re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper() == "T"


def _build_faces(entities: dict) -> dict[str, _Face]:
    faces: dict[str, _Face] = {}
    for face_id, entity in entities.items():
        if entity.get("type") != "ADVANCED_FACE":
            continue
        refs = _refs(entity.get("args", ""))
        surface_id = next(
            (ref for ref in reversed(refs) if _entity_type(entities, ref) in _SURFACE_TYPES),
            "",
        )
        if not surface_id:
            continue
        bounds: list[tuple[str, set[str]]] = []
        for ref in refs:
            bound_type = _entity_type(entities, ref)
            if bound_type not in _BOUND_TYPES:
                continue
            kind = "outer" if bound_type == "FACE_OUTER_BOUND" else "inner"
            bounds.append((kind, _resolve_edge_curves(ref, entities)))
        faces[face_id] = _Face(
            face_id=face_id,
            surface_id=surface_id,
            surface_type=_entity_type(entities, surface_id),
            same_sense=_face_same_sense(entity.get("args", "")),
            bounds=bounds,
        )
    return faces


def _edge_geometry_type(edge_id: str, entities: dict) -> str:
    entity = entities.get(edge_id) or {}
    for ref in _refs(entity.get("args", "")):
        entity_type = _entity_type(entities, ref)
        if entity_type in {"CIRCLE", "LINE", "ELLIPSE", "B_SPLINE_CURVE_WITH_KNOTS"}:
            return entity_type
    return ""


def _point_from_ref(ref: str, entities: dict) -> tuple[float, float, float] | None:
    entity = entities.get(ref) or {}
    entity_type = entity.get("type")
    if entity_type == "CARTESIAN_POINT":
        return _vector(entity.get("args", ""))
    if entity_type == "VERTEX_POINT":
        for child in _refs(entity.get("args", "")):
            point = _point_from_ref(child, entities)
            if point is not None:
                return point
    return None


def _edge_points(edge_id: str, entities: dict) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for ref in _refs((entities.get(edge_id) or {}).get("args", "")):
        if _entity_type(entities, ref) != "VERTEX_POINT":
            continue
        point = _point_from_ref(ref, entities)
        if point is not None:
            points.append(point)
    return points


def _surface_frame(
    surface_id: str, entities: dict
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    surface = entities.get(surface_id) or {}
    placement_id = next(
        (ref for ref in _refs(surface.get("args", ""))
         if _entity_type(entities, ref) == "AXIS2_PLACEMENT_3D"),
        "",
    )
    placement = entities.get(placement_id) or {}
    point = None
    direction = None
    for ref in _refs(placement.get("args", "")):
        entity_type = _entity_type(entities, ref)
        if entity_type == "CARTESIAN_POINT" and point is None:
            point = _point_from_ref(ref, entities)
        elif entity_type == "DIRECTION" and direction is None:
            direction = _normalize(_vector((entities.get(ref) or {}).get("args", "")) or (0, 0, 0))
    if point is None or direction is None:
        return None
    return point, direction


def _cylinder_radius(surface_id: str, entities: dict) -> float | None:
    values = _numbers((entities.get(surface_id) or {}).get("args", ""))
    if not values:
        return None
    radius = float(values[-1])
    return radius if math.isfinite(radius) and radius > 1e-8 else None


def _edge_face_map(faces: dict[str, _Face]) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {}
    for face in faces.values():
        for kind, edges in face.bounds:
            for edge_id in edges:
                result.setdefault(edge_id, []).append((face.face_id, kind))
    return result


def _axial_interval(
    edge_ids: Iterable[str], axis_point, axis_direction, entities: dict
) -> tuple[float, float] | None:
    values: list[float] = []
    for edge_id in edge_ids:
        for point in _edge_points(edge_id, entities):
            values.append(_dot(_sub(point, axis_point), axis_direction))
    if not values:
        return None
    return min(values), max(values)


def _same_axis(a: _Cylinder, b: _Cylinder, tolerance: float) -> bool:
    if abs(_dot(a.axis_direction, b.axis_direction)) < 1.0 - 1e-5:
        return False
    line_distance = _norm(_cross(_sub(b.axis_point, a.axis_point), a.axis_direction))
    if line_distance > tolerance:
        return False
    if a.axial_interval is None or b.axial_interval is None:
        return True
    a0, a1 = a.axial_interval
    # Express b's interval in a's axis coordinate system.
    offset = _dot(_sub(b.axis_point, a.axis_point), a.axis_direction)
    sign = 1.0 if _dot(a.axis_direction, b.axis_direction) >= 0 else -1.0
    b_values = (offset + sign * b.axial_interval[0], offset + sign * b.axial_interval[1])
    b0, b1 = min(b_values), max(b_values)
    return not (a1 < b0 - tolerance or b1 < a0 - tolerance)


def _group_cylinders(cylinders: list[_Cylinder], tolerance: float) -> list[list[_Cylinder]]:
    groups: list[list[_Cylinder]] = []
    for cylinder in cylinders:
        matching = [group for group in groups if any(_same_axis(cylinder, item, tolerance) for item in group)]
        if not matching:
            groups.append([cylinder])
            continue
        target = matching[0]
        target.append(cylinder)
        for extra in matching[1:]:
            target.extend(extra)
            groups.remove(extra)
    return groups


def _orthogonal_basis(normal):
    reference = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.8 else (0.0, 1.0, 0.0)
    u = _normalize(_cross(normal, reference)) or (0.0, 0.0, 1.0)
    v = _normalize(_cross(normal, u)) or (0.0, 1.0, 0.0)
    return u, v


def _recognize_slots(
    entities: dict,
    faces: dict[str, _Face],
    edge_faces: dict[str, list[tuple[str, str]]],
    tolerance: float,
) -> tuple[int, int, float]:
    """Recognize conservative recessed planar regions and classify by aspect ratio."""
    vertex_points = [
        point
        for ref, entity in entities.items()
        if entity.get("type") == "VERTEX_POINT"
        for point in [_point_from_ref(ref, entities)]
        if point is not None
    ]
    if len(vertex_points) < 4:
        return 0, 0, 0.0

    slots = pockets = 0
    ratios: list[float] = []
    for face in faces.values():
        if face.surface_type != "PLANE":
            continue
        frame = _surface_frame(face.surface_id, entities)
        if frame is None:
            continue
        origin, normal = frame
        projections = [_dot(point, normal) for point in vertex_points]
        plane_position = _dot(origin, normal)
        span = max(projections) - min(projections)
        margin = max(tolerance * 5.0, span * 0.01)
        # Exterior support planes are not recess bottoms.
        if plane_position <= min(projections) + margin or plane_position >= max(projections) - margin:
            continue

        adjacent_faces = {
            other_face_id
            for edge_id in face.edges
            for other_face_id, _ in edge_faces.get(edge_id, [])
            if other_face_id != face.face_id
        }
        if len(adjacent_faces) < 3:
            continue

        boundary_points = [
            point
            for edge_id in face.edges
            for point in _edge_points(edge_id, entities)
        ]
        if len(boundary_points) < 3:
            continue
        u, v = _orthogonal_basis(normal)
        us = [_dot(_sub(point, origin), u) for point in boundary_points]
        vs = [_dot(_sub(point, origin), v) for point in boundary_points]
        width = max(us) - min(us)
        height = max(vs) - min(vs)
        short = min(width, height)
        long = max(width, height)
        if short <= tolerance or long <= tolerance:
            continue
        ratio = long / short
        ratios.append(ratio)
        if ratio >= 2.0:
            slots += 1
        else:
            pockets += 1

    average_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return slots, pockets, average_ratio


def recognize_features_v11(entities: dict, info: dict) -> dict[str, float | int] | None:
    """Return replacements for existing manufacturing fields, or ``None``.

    No keys outside the existing API contract are returned.
    """
    faces = _build_faces(entities)
    if not faces:
        return None
    edge_faces = _edge_face_map(faces)
    diagonal = float(info.get("bbox_diagonal") or 0.0)
    tolerance = max(diagonal * 1e-4, 1e-5)

    cylinders: list[_Cylinder] = []
    for face in faces.values():
        if face.surface_type != "CYLINDRICAL_SURFACE":
            continue
        frame = _surface_frame(face.surface_id, entities)
        radius = _cylinder_radius(face.surface_id, entities)
        if frame is None or radius is None:
            continue
        axis_point, axis_direction = frame
        circle_edges = {
            edge_id for edge_id in face.edges
            if _edge_geometry_type(edge_id, entities) == "CIRCLE"
        }

        opening_evidence = False
        for edge_id in circle_edges:
            for adjacent_face_id, adjacent_bound_kind in edge_faces.get(edge_id, []):
                if adjacent_face_id == face.face_id:
                    continue
                adjacent = faces.get(adjacent_face_id)
                if adjacent and adjacent.surface_type == "PLANE" and adjacent_bound_kind == "inner":
                    opening_evidence = True
                    break

        # Same-sense false is the STEP semantic for an inward-facing cylinder
        # in a normally oriented solid.  Inner-bound evidence handles exporters
        # whose face orientation metadata is less consistent.
        inward = face.same_sense is False
        if not inward and not opening_evidence:
            continue
        if not circle_edges:
            continue
        cylinders.append(
            _Cylinder(
                face_id=face.face_id,
                axis_point=axis_point,
                axis_direction=axis_direction,
                radius=radius,
                edge_ids=face.edges,
                circle_edges=circle_edges,
                axial_interval=_axial_interval(face.edges, axis_point, axis_direction, entities),
                inward_facing=inward,
            )
        )

    through_holes = blind_holes = 0
    hole_depths: list[float] = []
    hole_aspects: list[float] = []
    for group in _group_cylinders(cylinders, tolerance):
        opening_edges: set[str] = set()
        closed_edges: set[str] = set()
        circle_edges = {edge_id for item in group for edge_id in item.circle_edges}
        for edge_id in circle_edges:
            for adjacent_face_id, bound_kind in edge_faces.get(edge_id, []):
                adjacent = faces.get(adjacent_face_id)
                if not adjacent or adjacent.surface_type != "PLANE":
                    continue
                if any(adjacent_face_id == item.face_id for item in group):
                    continue
                if bound_kind == "inner":
                    opening_edges.add(edge_id)
                elif bound_kind == "outer":
                    closed_edges.add(edge_id)

        # One physical hole may contain several coaxial cylindrical faces
        # (counterbore, split wall, chamfer transition).  Count the group once.
        if len(opening_edges) >= 2:
            through_holes += 1
        elif opening_edges or closed_edges:
            blind_holes += 1
        elif len(circle_edges) >= 2 and any(item.inward_facing for item in group):
            # Topology is incomplete but the closed inward cylinder is still a
            # credible hole.  Prefer through only when both terminal circles exist.
            through_holes += 1
        else:
            continue

        intervals = [item.axial_interval for item in group if item.axial_interval is not None]
        if intervals:
            depth = max(end for _, end in intervals) - min(start for start, _ in intervals)
            if depth > tolerance:
                hole_depths.append(depth)
                radius = min(item.radius for item in group)
                hole_aspects.append(depth / max(2.0 * radius, tolerance))

    slots, pockets, average_slot_ratio = _recognize_slots(
        entities, faces, edge_faces, tolerance
    )
    scale = max(float(max(info.get("bbox_dims") or [0.0])), tolerance)
    return {
        "through_holes": through_holes,
        "blind_holes": blind_holes,
        "avg_hole_depth_ratio": min(
            (sum(hole_depths) / len(hole_depths)) / scale, 1.0
        ) if hole_depths else 0.0,
        "hole_aspect_ratio": min(
            (sum(hole_aspects) / len(hole_aspects)) / 5.0, 1.0
        ) if hole_aspects else 0.0,
        "slots": slots,
        "pockets": pockets,
        "avg_slot_lw_ratio": min(average_slot_ratio / 5.0, 1.0),
    }
