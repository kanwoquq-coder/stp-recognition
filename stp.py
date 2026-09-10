# stp.py
"""
STP 零件相似度检索系统 - 大模型版（无CAD内核依赖）
支持 OpenAI 兼容接口（深度求索/智谱/通义/中转站等）
"""

import re
import os
import json
import time
import math
import base64
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from openai import OpenAI
import chromadb
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from recall_debugger import RecallDebugger

# 可选FAISS向量索引模块
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[警告] FAISS未安装，将使用ChromaDB作为向量索引")

# 可选渲染模块（如果安装了pyvista和pythonocc-core）
try:
    import pyvista as pv
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    import imageio  # 用于生成旋转动画GIF
    HAS_RENDER = True
except ImportError:
    HAS_RENDER = False

# ============================================================
# 第一部分：STP 深度文本解析
# ============================================================

_CARTESIAN_COORD_RE = re.compile(
    r'\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,'
    r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,'
    r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)'
)


def _parse_cartesian_coords(args: str) -> Optional[List[float]]:
    """从 CARTESIAN_POINT 参数中解析 (x, y, z)。"""
    coord_match = _CARTESIAN_COORD_RE.search(args)
    if not coord_match:
        return None
    try:
        return [
            float(coord_match.group(1)),
            float(coord_match.group(2)),
            float(coord_match.group(3)),
        ]
    except ValueError:
        return None


def _extract_solid_points(entities: dict) -> Tuple[List[List[float]], str, int]:
    """
    提取用于包围盒/点分布的实体点。

    优先使用 VERTEX_POINT 引用的 CARTESIAN_POINT，避免构造线、
    辅助坐标系等非实体坐标污染 bbox（例如测试件3被拉长到129mm）。
    若顶点不足则回退到全部 CARTESIAN_POINT。

    Returns:
        (points, source, num_cartesian_total)
        source: 'vertex' | 'cartesian_fallback' | 'none'
    """
    cartesian: Dict[str, List[float]] = {}
    for eid, e in entities.items():
        if e['type'] == 'CARTESIAN_POINT':
            coords = _parse_cartesian_coords(e['args'])
            if coords is not None:
                cartesian[eid] = coords

    vertex_points: List[List[float]] = []
    for e in entities.values():
        if e['type'] != 'VERTEX_POINT':
            continue
        refs = re.findall(r'#\d+', e['args'])
        for ref in refs:
            if ref in cartesian:
                vertex_points.append(cartesian[ref])
                break

    if len(vertex_points) >= 2:
        return vertex_points, 'vertex', len(cartesian)

    all_points = list(cartesian.values())
    if all_points:
        return all_points, 'cartesian_fallback', len(cartesian)
    return [], 'none', 0


def parse_stp_deep(filepath: str) -> dict:
    content = Path(filepath).read_text(errors='ignore')
    info = {'filename': Path(filepath).name, 'filepath': filepath}

    # 解析所有实体
    entities = {}
    flat = re.sub(r'\n\s*', ' ', content)
    for m in re.finditer(r'(#\d+)\s*=\s*([A-Z_]+)\s*\(([^;]*)\)\s*;', flat):
        eid, etype, eargs = m.group(1), m.group(2), m.group(3).strip()
        entities[eid] = {'type': etype, 'args': eargs}

    type_counts = {}
    for e in entities.values():
        type_counts[e['type']] = type_counts.get(e['type'], 0) + 1

    info['entity_counts'] = type_counts
    info['total_entities'] = len(entities)

    # 拓扑
    info['num_faces'] = type_counts.get('ADVANCED_FACE', 0)
    info['num_edges'] = type_counts.get('EDGE_CURVE', 0)
    info['num_vertices'] = type_counts.get('VERTEX_POINT', 0)
    info['num_edge_loops'] = type_counts.get('EDGE_LOOP', 0)
    info['num_oriented_edges'] = type_counts.get('ORIENTED_EDGE', 0)
    info['num_closed_shells'] = type_counts.get('CLOSED_SHELL', 0)
    info['num_solids'] = type_counts.get('MANIFOLD_SOLID_BREP', 0)
    info['euler'] = info['num_vertices'] - info['num_edges'] + info['num_faces']

    # 面类型
    surface_types = {
        'PLANE': 'plane', 'CYLINDRICAL_SURFACE': 'cylinder',
        'CONICAL_SURFACE': 'cone', 'SPHERICAL_SURFACE': 'sphere',
        'TOROIDAL_SURFACE': 'torus', 'B_SPLINE_SURFACE_WITH_KNOTS': 'bspline',
    }
    face_type_counts = {}
    for eid, e in entities.items():
        if e['type'] == 'ADVANCED_FACE':
            refs = re.findall(r'#\d+', e['args'])
            if refs:
                surf_ref = refs[-1]
                if surf_ref in entities:
                    stype = entities[surf_ref]['type']
                    label = surface_types.get(stype, 'other')
                    face_type_counts[label] = face_type_counts.get(label, 0) + 1
    if not face_type_counts:
        for stype, label in surface_types.items():
            count = type_counts.get(stype, 0)
            if count > 0:
                face_type_counts[label] = count
    info['face_types'] = face_type_counts

    # 边类型
    curve_types = {
        'LINE': 'line', 'CIRCLE': 'circle',
        'ELLIPSE': 'ellipse', 'B_SPLINE_CURVE_WITH_KNOTS': 'bspline',
    }
    edge_type_counts = {}
    for eid, e in entities.items():
        if e['type'] == 'EDGE_CURVE':
            refs = re.findall(r'#\d+', e['args'])
            for ref in refs:
                if ref in entities and entities[ref]['type'] in curve_types:
                    label = curve_types[entities[ref]['type']]
                    edge_type_counts[label] = edge_type_counts.get(label, 0) + 1
                    break
    if not edge_type_counts:
        for ctype, label in curve_types.items():
            count = type_counts.get(ctype, 0)
            if count > 0:
                edge_type_counts[label] = count
    info['edge_types'] = edge_type_counts

    # 坐标点：包围盒优先用 VERTEX_POINT，避免辅助几何污染
    points, bbox_source, num_cartesian = _extract_solid_points(entities)
    info['num_points'] = num_cartesian
    info['num_bbox_points'] = len(points)
    info['bbox_source'] = bbox_source

    if points:
        pts = np.array(points)
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        dims = sorted([
            round(maxs[0] - mins[0], 4),
            round(maxs[1] - mins[1], 4),
            round(maxs[2] - mins[2], 4),
        ])
        info['bbox_dims'] = dims
        info['bbox_diagonal'] = round(math.sqrt(sum(d**2 for d in dims)), 4)
        centered = pts - pts.mean(axis=0)
        if len(centered) > 3:
            cov = np.cov(centered.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            if eigvals[0] > 1e-12:
                info['point_distribution_ratios'] = [round(v / eigvals[0], 4) for v in eigvals]
            else:
                info['point_distribution_ratios'] = [1.0, 1.0, 1.0]
        else:
            info['point_distribution_ratios'] = [1.0, 1.0, 1.0]
    else:
        info['bbox_dims'] = [0, 0, 0]
        info['bbox_diagonal'] = 0
        info['point_distribution_ratios'] = [1.0, 1.0, 1.0]

    # 半径
    radii = []
    for eid, e in entities.items():
        if e['type'] == 'CIRCLE':
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', e['args'])
            if nums:
                try:
                    r = float(nums[-1])
                    if 0.001 < r < 100000:
                        radii.append(round(r, 4))
                except ValueError:
                    pass
    info['radii'] = radii
    info['unique_radii'] = sorted(set(round(r, 2) for r in radii))
    info['num_unique_radii'] = len(info['unique_radii'])

    cyl_radii = []
    for eid, e in entities.items():
        if e['type'] == 'CYLINDRICAL_SURFACE':
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', e['args'])
            if nums:
                try:
                    r = float(nums[-1])
                    if 0.001 < r < 100000:
                        cyl_radii.append(round(r, 4))
                except ValueError:
                    pass
    info['cylinder_radii'] = sorted(set(round(r, 2) for r in cyl_radii))

    # 无量纲指标
    dims = info['bbox_dims']
    if dims[2] > 1e-6:
        info['aspect_ratios'] = [round(dims[0] / dims[2], 4), round(dims[1] / dims[2], 4)]
    else:
        info['aspect_ratios'] = [0, 0]

    total_f = max(info['num_faces'], 1)
    info['face_area_cv_proxy'] = round(len(info['unique_radii']) / total_f, 4)

    ft_vals = list(face_type_counts.values())
    ft_total = sum(ft_vals) if ft_vals else 1
    entropy = 0
    for v in ft_vals:
        p = v / ft_total
        if p > 0:
            entropy -= p * math.log2(p)
    info['face_type_entropy'] = round(entropy, 4)

    # 新增几何特征
    info['total_surface_area_estimate'] = round(info['bbox_diagonal'] ** 2 * info['num_faces'] * 0.1, 4)
    info['volume_estimate'] = round(dims[0] * dims[1] * dims[2] * 0.3, 4)
    info['compactness'] = round(info['volume_estimate'] / (info['total_surface_area_estimate'] + 1e-10), 4)

    # 复杂度指标
    info['edge_face_ratio'] = round(info['num_edges'] / max(info['num_faces'], 1), 4)
    info['vertex_face_ratio'] = round(info['num_vertices'] / max(info['num_faces'], 1), 4)
    info['curve_complexity'] = round((edge_type_counts.get('bspline', 0) + edge_type_counts.get('ellipse', 0)) / max(total_e := max(info['num_edges'], 1), 1), 4)

    # 特征丰富度
    info['feature_richness'] = round(len(info['unique_radii']) + len(info['cylinder_radii']), 4)
    info['rotational_symmetry_score'] = round(cyl_ratio := (info['face_types'].get('cylinder', 0) / total_f) * (1 - abs(dims[0] - dims[1]) / (dims[2] + 1e-10)), 4)

    # 制造特征识别
    try:
        mfg_result = extract_manufacturing_features_standalone(entities, info)
        info['mfg_features'] = mfg_result
        info['mfg_feature_vector'] = mfg_result['mfg_feature_vector']
    except Exception:
        # 制造特征识别失败时，设置默认值
        info['mfg_features'] = {
            'through_holes': 0, 'blind_holes': 0, 'slots': 0, 'pockets': 0,
            'bosses': 0, 'ribs': 0, 'chamfers': 0, 'fillets': 0,
            'threads': 0, 'has_thread': 0, 'complexity_score': 0,
            'mfg_feature_vector': np.zeros(32, dtype=np.float32)
        }
        info['mfg_feature_vector'] = info['mfg_features']['mfg_feature_vector']

    return info


# ============================================================
# 第二部分：描述生成
# ============================================================

def generate_description(info: dict) -> str:
    """
    生成零件描述（精简版）
    只保留：尺寸、孔洞大小、孔洞数、长宽比
    """
    lines = []
    dims = info['bbox_dims']
    ar = info['aspect_ratios']

    # 1. 零件名称
    lines.append(f"零件名称: {info['filename']}")

    # 2. 尺寸信息
    lines.append(f"尺寸: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} mm")

    # 3. 长宽比
    lines.append(f"长宽比: {ar[0]:.2f}, 高宽比: {ar[1]:.2f}")

    # 4. 孔洞信息
    hole_count = info['num_unique_radii']
    if hole_count > 0:
        lines.append(f"孔洞数: {hole_count}")
        # 孔洞大小（半径）
        radii = info.get('unique_radii', [])
        if radii:
            radii_str = ', '.join([f"{r:.2f}" for r in radii[:5]])  # 最多显示5个
            if len(radii) > 5:
                radii_str += f"... 共{len(radii)}个"
            lines.append(f"孔洞半径: {radii_str} mm")
    else:
        lines.append("孔洞数: 0")

    # 5. 制造特征信息
    mfg = info.get('mfg_features', {})
    if mfg:
        mfg_parts = []
        if mfg.get('through_holes', 0) > 0:
            mfg_parts.append(f"通孔:{mfg['through_holes']}")
        if mfg.get('blind_holes', 0) > 0:
            mfg_parts.append(f"盲孔:{mfg['blind_holes']}")
        if mfg.get('slots', 0) > 0:
            mfg_parts.append(f"槽:{mfg['slots']}")
        if mfg.get('pockets', 0) > 0:
            mfg_parts.append(f"型腔:{mfg['pockets']}")
        if mfg.get('bosses', 0) > 0:
            mfg_parts.append(f"凸台:{mfg['bosses']}")
        if mfg.get('ribs', 0) > 0:
            mfg_parts.append(f"筋:{mfg['ribs']}")
        if mfg.get('fillets', 0) > 0:
            mfg_parts.append(f"圆角:{mfg['fillets']}")
        if mfg.get('chamfers', 0) > 0:
            mfg_parts.append(f"倒角:{mfg['chamfers']}")
        if mfg.get('has_thread', 0) > 0:
            mfg_parts.append("有螺纹")
        if mfg_parts:
            lines.append(f"制造特征: {'; '.join(mfg_parts)}")

    return '\n'.join(lines)


def generate_retrieval_description(info: dict) -> str:
    """生成无文件名/料号的形状描述，专用于文本向量召回。"""
    dims = info.get('bbox_dims', [0, 0, 0])
    ar = info.get('aspect_ratios', [0, 0])
    face_types = info.get('face_types', {})
    edge_types = info.get('edge_types', {})
    radii = info.get('unique_radii', [])
    mfg = info.get('mfg_features', {})
    return '\n'.join([
        f"尺寸: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} mm",
        f"包围盒比例: {ar[0]:.4f}, {ar[1]:.4f}",
        f"拓扑: 面{info.get('num_faces', 0)}, 边{info.get('num_edges', 0)}, "
        f"顶点{info.get('num_vertices', 0)}, 欧拉数{info.get('euler', 0)}",
        "面类型: " + ', '.join(f"{k}={v}" for k, v in sorted(face_types.items())),
        "边类型: " + ', '.join(f"{k}={v}" for k, v in sorted(edge_types.items())),
        "圆特征半径: " + (', '.join(f"{r:.2f}" for r in radii) if radii else '无'),
        f"制造结构代理: 通孔{mfg.get('through_holes', 0)}, "
        f"盲孔{mfg.get('blind_holes', 0)}, 凸台{mfg.get('bosses', 0)}, "
        f"筋{mfg.get('ribs', 0)}, 槽{mfg.get('slots', 0)}",
    ])


# ============================================================
# 第三部分：传统几何特征相似度计算
# ============================================================

def calculate_geometric_similarity(info1: dict, info2: dict) -> dict:
    """计算两个STP文件的几何特征相似度"""
    similarities = {}

    # 面类型分布相似度
    ft1 = info1['face_types']
    ft2 = info2['face_types']
    total_f1 = max(info1['num_faces'], 1)
    total_f2 = max(info2['num_faces'], 1)

    face_types = ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'bspline', 'other']
    ratios1 = [ft1.get(t, 0) / total_f1 for t in face_types]
    ratios2 = [ft2.get(t, 0) / total_f2 for t in face_types]
    face_sim = 1 - sum(abs(r1 - r2) for r1, r2 in zip(ratios1, ratios2)) / 2
    similarities['face_type_distribution'] = round(face_sim, 4)

    # 拓扑相似度
    topo_features = ['num_faces', 'num_edges', 'num_vertices', 'num_edge_loops', 'euler']
    topo_sim = 0
    for feat in topo_features:
        val1 = info1.get(feat, 0)
        val2 = info2.get(feat, 0)
        max_val = max(val1, val2, 1)
        topo_sim += 1 - abs(val1 - val2) / max_val
    similarities['topology'] = round(topo_sim / len(topo_features), 4)

    # 尺寸比例相似度
    ar1 = info1['aspect_ratios']
    ar2 = info2['aspect_ratios']
    ar_sim = 1 - (abs(ar1[0] - ar2[0]) + abs(ar1[1] - ar2[1])) / 2
    similarities['aspect_ratio'] = round(max(ar_sim, 0), 4)

    # 点分布相似度
    pd1 = info1['point_distribution_ratios']
    pd2 = info2['point_distribution_ratios']
    pd_sim = 1 - sum(abs(p1 - p2) for p1, p2 in zip(pd1, pd2)) / 3
    similarities['point_distribution'] = round(max(pd_sim, 0), 4)

    # 复杂度相似度
    entropy_sim = 1 - abs(info1['face_type_entropy'] - info2['face_type_entropy']) / max(info1['face_type_entropy'], info2['face_type_entropy'], 1)
    similarities['complexity'] = round(max(entropy_sim, 0), 4)

    # 特征丰富度相似度
    fr_sim = 1 - abs(info1['feature_richness'] - info2['feature_richness']) / max(info1['feature_richness'], info2['feature_richness'], 1)
    similarities['feature_richness'] = round(max(fr_sim, 0), 4)

    # 紧密度相似度
    compact_sim = 1 - abs(info1['compactness'] - info2['compactness']) / max(info1['compactness'], info2['compactness'], 1)
    similarities['compactness'] = round(max(compact_sim, 0), 4)

    # 旋转对称性相似度
    rot_sim = 1 - abs(info1['rotational_symmetry_score'] - info2['rotational_symmetry_score']) / max(abs(info1['rotational_symmetry_score']), abs(info2['rotational_symmetry_score']), 1)
    similarities['rotational_symmetry'] = round(max(rot_sim, 0), 4)

    # 综合相似度（加权）- 包含制造特征
    weights = {
        'face_type_distribution': 0.20,
        'topology': 0.15,
        'aspect_ratio': 0.10,
        'point_distribution': 0.08,
        'complexity': 0.08,
        'feature_richness': 0.08,
        'compactness': 0.05,
        'rotational_symmetry': 0.05
    }
    # 原权重总和为0.79；归一化后完全相同零件的基础分才是1.0
    weight_total = sum(weights.values())
    overall_sim = sum(similarities[key] * weights[key] for key in weights) / weight_total
    similarities['overall'] = round(overall_sim, 4)

    # 制造特征相似度（如果可用）
    mfg1 = info1.get('mfg_features')
    mfg2 = info2.get('mfg_features')
    if mfg1 and mfg2:
        mfg_sim = calculate_mfg_similarity(mfg1, mfg2)
        similarities['manufacturing'] = mfg_sim
        # 将制造特征以10%权重融合到综合评分中
        similarities['overall'] = round(overall_sim * 0.9 + mfg_sim['overall'] * 0.1, 4)

    return similarities


# ============================================================
# 独立的几何/视觉向量提取（不依赖backend模块）
# ============================================================

def extract_geo_vector_standalone(info: dict) -> np.ndarray:
    """
    独立的几何特征向量提取（64维）
    不依赖backend模块，直接从parse_stp_deep结果提取
    """
    vector = np.zeros(64, dtype=np.float32)
    idx = 0

    # 1. 尺寸特征 (12维)
    dims = info.get('bbox_dims', [0, 0, 0])
    diagonal = info.get('bbox_diagonal', 1)

    # 归一化尺寸
    for d in dims:
        vector[idx] = min(d / max(diagonal, 1), 1.0)
        idx += 1

    # 尺寸比例
    if dims[2] > 0:
        vector[idx] = dims[0] / dims[2]  # 长宽比
        idx += 1
        vector[idx] = dims[1] / dims[2]  # 高宽比
        idx += 1
    else:
        idx += 2

    # 对角线对数
    vector[idx] = min(math.log10(max(diagonal, 1)) / 5, 1.0)
    idx += 1

    # 体积估计对数
    volume = dims[0] * dims[1] * dims[2]
    vector[idx] = min(math.log10(max(volume, 1)) / 6, 1.0)
    idx += 1

    # 点分布特征 (3维)
    pd = info.get('point_distribution_ratios', [1, 1, 1])
    for p in pd[:3]:
        vector[idx] = p
        idx += 1

    # 紧密度
    vector[idx] = info.get('compactness', 0)
    idx += 1

    # 2. 面类型分布 (7维)
    ft = info.get('face_types', {})
    total_f = max(info.get('num_faces', 1), 1)
    for face_type in ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'bspline', 'other']:
        vector[idx] = ft.get(face_type, 0) / total_f
        idx += 1

    # 3. 拓扑特征 (5维)
    vector[idx] = min(math.log10(max(info.get('num_faces', 1), 1)) / 3, 1.0)
    idx += 1
    vector[idx] = min(math.log10(max(info.get('num_edges', 1), 1)) / 3, 1.0)
    idx += 1
    vector[idx] = min(math.log10(max(info.get('num_vertices', 1), 1)) / 3, 1.0)
    idx += 1
    vector[idx] = (info.get('euler', 0) + 10) / 20  # 归一化欧拉数
    idx += 1
    vector[idx] = info.get('edge_face_ratio', 0) / 5
    idx += 1

    # 4. 形状特征 (6维)
    ar = info.get('aspect_ratios', [0, 0])
    vector[idx] = ar[0] if ar else 0
    idx += 1
    vector[idx] = ar[1] if len(ar) > 1 else 0
    idx += 1

    # 延伸度
    vector[idx] = 1 - min(dims[0] / max(dims[2], 0.001), 1.0)
    idx += 1

    # 扁平度
    vector[idx] = 1 - min(dims[1] / max(dims[2], 0.001), 1.0)
    idx += 1

    # 球形度
    vector[idx] = info.get('compactness', 0) * 2
    idx += 1

    # 矩形度
    vector[idx] = 1 - info.get('face_type_entropy', 0) / 3
    idx += 1

    # 5. 复杂度特征 (6维)
    vector[idx] = info.get('face_type_entropy', 0) / 3
    idx += 1
    vector[idx] = min(info.get('feature_richness', 0) / 20, 1.0)
    idx += 1
    vector[idx] = info.get('rotational_symmetry_score', 0)
    idx += 1
    vector[idx] = info.get('curve_complexity', 0)
    idx += 1
    vector[idx] = min(info.get('num_unique_radii', 0) / 10, 1.0)
    idx += 1
    vector[idx] = min(len(info.get('cylinder_radii', [])) / 10, 1.0)
    idx += 1

    # 6. 边类型分布 (4维)
    et = info.get('edge_types', {})
    total_e = max(info.get('num_edges', 1), 1)
    for edge_type in ['line', 'circle', 'ellipse', 'bspline']:
        vector[idx] = et.get(edge_type, 0) / total_e
        idx += 1

    # 7. 孔特征 (6维) - 增强: 使用制造特征中的通孔/盲孔数据
    mfg = info.get('mfg_features', {})
    total_holes_mfg = (mfg.get('through_holes', 0) + mfg.get('blind_holes', 0)) if mfg else 0

    # 通孔标志 (增强: 使用制造特征数据)
    if total_holes_mfg > 0:
        through_ratio = mfg.get('through_holes', 0) / max(total_holes_mfg, 1)
        vector[idx] = through_ratio  # 通孔比例，更精确
    else:
        vector[idx] = 1.0 if info.get('euler', 0) < 2 else 0.0  # 回退到欧拉数判断
    idx += 1

    # 孔数量估计
    vector[idx] = min(info.get('num_unique_radii', 0) / 10, 1.0)
    idx += 1

    # 平均半径 (增强: 结合通孔/盲孔比例)
    radii = info.get('unique_radii', [])
    if radii:
        vector[idx] = min(sum(radii) / len(radii) / 10, 1.0)
    else:
        vector[idx] = 0
    if total_holes_mfg > 0:
        # 轻微调整: 通孔比例高的零件, 平均半径通常较大
        through_ratio = mfg.get('through_holes', 0) / max(total_holes_mfg, 1)
        vector[idx] = min(vector[idx] * (0.8 + 0.2 * through_ratio), 1.0)
    idx += 1

    # 半径方差 (增强: 结合盲孔比例)
    if len(radii) > 1:
        mean_r = sum(radii) / len(radii)
        var_r = sum((r - mean_r) ** 2 for r in radii) / len(radii)
        vector[idx] = min(math.sqrt(var_r) / 5, 1.0)
    else:
        vector[idx] = 0
    if total_holes_mfg > 0:
        # 盲孔比例高的零件, 半径方差通常更大(不同规格的孔)
        blind_ratio = mfg.get('blind_holes', 0) / max(total_holes_mfg, 1)
        vector[idx] = min(vector[idx] * (0.8 + 0.4 * blind_ratio), 1.0)
    idx += 1

    # 圆柱面数量
    vector[idx] = min(len(info.get('cylinder_radii', [])) / 10, 1.0)
    idx += 1

    # 圆柱面比例
    vector[idx] = ft.get('cylinder', 0) / total_f
    idx += 1

    # 8. 形状类型编码 (13维)
    cyl_ratio = ft.get('cylinder', 0) / total_f if total_f > 0 else 0
    plane_ratio = ft.get('plane', 0) / total_f if total_f > 0 else 0
    bspline_ratio = ft.get('bspline', 0) / total_f if total_f > 0 else 0
    sphere_ratio = ft.get('sphere', 0) / total_f if total_f > 0 else 0
    cone_ratio = ft.get('cone', 0) / total_f if total_f > 0 else 0
    torus_ratio = ft.get('torus', 0) / total_f if total_f > 0 else 0

    # 是否旋转体
    vector[idx] = 1.0 if cyl_ratio > 0.5 else 0.0
    idx += 1

    # 是否棱柱体
    vector[idx] = 1.0 if plane_ratio > 0.6 else 0.0
    idx += 1

    # 是否自由曲面
    vector[idx] = 1.0 if bspline_ratio > 0.3 else 0.0
    idx += 1

    # 是否球体
    vector[idx] = 1.0 if sphere_ratio > 0.3 else 0.0
    idx += 1

    # 是否圆锥体
    vector[idx] = 1.0 if cone_ratio > 0.3 else 0.0
    idx += 1

    # 是否有圆角
    vector[idx] = 1.0 if torus_ratio > 0.05 else 0.0
    idx += 1

    # 是否混合类型
    type_count = sum(1 for r in [cyl_ratio, plane_ratio, bspline_ratio, sphere_ratio, cone_ratio] if r > 0.1)
    vector[idx] = 1.0 if type_count >= 3 else 0.0
    idx += 1

    # 面类型占比连续编码 (6维)
    vector[idx] = cyl_ratio
    idx += 1
    vector[idx] = plane_ratio
    idx += 1
    vector[idx] = bspline_ratio
    idx += 1
    vector[idx] = sphere_ratio
    idx += 1
    vector[idx] = cone_ratio
    idx += 1
    vector[idx] = torus_ratio
    idx += 1

    # 确保所有值在[0, 1]范围内
    vector = np.clip(vector, 0.0, 1.0)

    return vector


_CLIP_RUNTIME = None
_CLIP_LOAD_ATTEMPTED = False


def extract_visual_vector_standalone(view_paths: List[str]) -> np.ndarray:
    """
    视觉特征向量提取（512维）
    优先使用 CLIP 多视图 mean-pool；CLIP 不可用时回退 PIL 手工特征。
    """
    if not view_paths:
        return np.zeros(512, dtype=np.float32)

    # ---------- CLIP 多视图 mean-pool ----------
    # 模型只尝试加载一次；默认仅使用本地缓存，避免离线建库时每个零件都等待下载失败。
    global _CLIP_RUNTIME, _CLIP_LOAD_ATTEMPTED
    if not _CLIP_LOAD_ATTEMPTED:
        _CLIP_LOAD_ATTEMPTED = True
        try:
            _clip_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.clip_cache')
            os.environ['TRANSFORMERS_CACHE'] = _clip_cache
            os.environ['HF_HOME'] = _clip_cache
            os.environ['TORCH_HOME'] = _clip_cache
            os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_clip_cache, 'ext')
            os.makedirs(os.path.join(_clip_cache, 'ext'), exist_ok=True)

            from transformers import CLIPProcessor, CLIPModel
            import torch
            local_only = os.environ.get('STP_CLIP_ALLOW_DOWNLOAD', '0') != '1'
            model_name = 'openai/clip-vit-base-patch32'
            processor = CLIPProcessor.from_pretrained(
                model_name, cache_dir=_clip_cache, local_files_only=local_only
            )
            model = CLIPModel.from_pretrained(
                model_name, cache_dir=_clip_cache, local_files_only=local_only
            )
            model.eval()
            _CLIP_RUNTIME = (processor, model, torch)
        except Exception:
            _CLIP_RUNTIME = None

    if _CLIP_RUNTIME is not None:
        try:
            from PIL import Image
            processor, model, torch = _CLIP_RUNTIME
            valid = [
                Image.open(path).convert('RGB')
                for path in view_paths[:6]
                if path and Path(path).exists()
            ]
            if valid:
                inputs = processor(images=valid, return_tensors='pt')
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
                vec = features.mean(dim=0).cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 1e-8 else vec
        except Exception:
            _CLIP_RUNTIME = None
    # ---------- PIL 轮廓特征（无 CLIP 时保底）----------
    # 旧实现直接对整张白底图缩放并平均，余弦相似度几乎由白色背景决定。
    # 新实现提取前景轮廓、裁边、居中，并把长边统一为水平方向；同时只平均有效视图。
    try:
        from PIL import Image

        def prepare_silhouette(path: str, size: int = 32) -> Optional[np.ndarray]:
            rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
            border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
            background = np.median(border, axis=0)
            delta = np.max(np.abs(rgb - background), axis=2)
            mask = delta > 12.0
            if int(mask.sum()) < max(16, int(mask.size * 0.002)):
                return None

            ys, xs = np.where(mask)
            crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            # 消除渲染坐标系造成的视图内90度旋转差异
            if crop.shape[0] > crop.shape[1]:
                crop = np.rot90(crop)

            h, w = crop.shape
            scale = (size - 4) / max(h, w)
            out_h = max(1, int(round(h * scale)))
            out_w = max(1, int(round(w * scale)))
            resampling = getattr(Image, 'Resampling', Image).LANCZOS
            resized = Image.fromarray((crop.astype(np.uint8) * 255)).resize(
                (out_w, out_h), resampling
            )
            canvas = np.zeros((size, size), dtype=np.float32)
            y0 = (size - out_h) // 2
            x0 = (size - out_w) // 2
            canvas[y0:y0 + out_h, x0:x0 + out_w] = (
                np.asarray(resized, dtype=np.float32) / 255.0
            )
            return canvas.reshape(-1)

        view_features = []
        for path in view_paths[:6]:
            if path and Path(path).exists():
                feature = prepare_silhouette(path)
                if feature is not None:
                    view_features.append(feature)

        if not view_features:
            return np.zeros(512, dtype=np.float32)

        avg = np.mean(view_features, axis=0)
        vec = avg.reshape(-1, 2).mean(axis=1).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec
    except Exception:
        return np.zeros(512, dtype=np.float32)

# ============================================================
# 制造特征识别增强
# ============================================================

def extract_manufacturing_features_standalone(entities: dict, info: dict) -> dict:
    """
    从STP实体结构中提取制造特征

    检测特征:
    - 通孔/盲孔: 通过CYLINDRICAL_SURFACE的拓扑关系分析
    - 槽/型腔: 通过面邻接关系检测
    - 凸台: 圆柱面超出包围盒检测
    - 加强筋: 高长宽比PLANE面检测
    - 倒角/圆角: TOROIDAL_SURFACE计数 + 面夹角分析
    - 螺纹: 小半径圆柱面 + B样条曲线密度检测

    Args:
        entities: 实体字典 {eid: {'type': ..., 'args': ...}}
        info: parse_stp_deep 返回的解析信息

    Returns:
        dict with keys:
            - mfg_details: 详细特征计数
            - mfg_feature_vector: 32维特征向量 (numpy.ndarray)
    """
    mfg = {
        'through_holes': 0,
        'blind_holes': 0,
        'avg_hole_depth_ratio': 0.0,
        'hole_aspect_ratio': 0.0,
        'slots': 0,
        'pockets': 0,
        'avg_slot_depth_ratio': 0.0,
        'avg_slot_lw_ratio': 0.0,
        'bosses': 0,
        'avg_boss_height_ratio': 0.0,
        'boss_density': 0.0,
        'ribs': 0,
        'rib_thickness_ratio': 0.0,
        'chamfers': 0,
        'fillets': 0,
        'avg_chamfer_angle': 0.0,
        'avg_fillet_radius_ratio': 0.0,
        'threads': 0,
        'has_thread': 0,
        'unique_feature_types': 0,
        'feature_density': 0.0,
        'feature_diversity': 0.0,
        'complexity_score': 0.0,
        'manufacturing_symmetry': 0.0,
    }

    # 收集实体引用关系
    # 建立面-曲面映射: ADVANCED_FACE -> surface reference
    face_to_surface = {}  # face_id -> surface_id
    surface_to_faces = {}  # surface_id -> [face_ids]
    for eid, e in entities.items():
        if e['type'] == 'ADVANCED_FACE':
            refs = re.findall(r'#\d+', e['args'])
            if refs:
                surf_ref = refs[-1]
                face_to_surface[eid] = surf_ref
                if surf_ref not in surface_to_faces:
                    surface_to_faces[surf_ref] = []
                surface_to_faces[surf_ref].append(eid)

    # 建立面-边映射: ADVANCED_FACE -> [edge_ids]
    face_to_edges = {}
    for eid, e in entities.items():
        if e['type'] == 'ADVANCED_FACE':
            refs = re.findall(r'#\d+', e['args'])
            if len(refs) >= 2:
                # 第一个ref通常是EDGE_LOOP或FACE_BOUND
                bound_ref = refs[0]
                if bound_ref in entities:
                    be = entities[bound_ref]
                    if be['type'] in ('EDGE_LOOP', 'FACE_BOUND'):
                        edge_refs = re.findall(r'#\d+', be['args'])
                        face_to_edges[eid] = edge_refs

    # 建立边-面映射: 共享EDGE_CURVE的面互为邻接
    edge_to_faces = {}
    for fid, edge_ids in face_to_edges.items():
        for eid in edge_ids:
            if eid not in edge_to_faces:
                edge_to_faces[eid] = []
            edge_to_faces[eid].append(fid)

    # 面邻接图
    face_adjacency = {}  # face_id -> set(adjacent_face_ids)
    for fid, edge_ids in face_to_edges.items():
        if fid not in face_adjacency:
            face_adjacency[fid] = set()
        for eid in edge_ids:
            adj_faces = edge_to_faces.get(eid, [])
            for af in adj_faces:
                if af != fid:
                    face_adjacency[fid].add(af)

    # 收集平面信息
    plane_faces = {}  # face_id -> surface_id (for PLANE surfaces)
    cyl_faces = {}    # face_id -> surface_id (for CYLINDRICAL_SURFACE)
    torus_faces = {}  # face_id -> surface_id (for TOROIDAL_SURFACE)

    for fid, surf_ref in face_to_surface.items():
        if surf_ref in entities:
            stype = entities[surf_ref]['type']
            if stype == 'PLANE':
                plane_faces[fid] = surf_ref
            elif stype == 'CYLINDRICAL_SURFACE':
                cyl_faces[fid] = surf_ref
            elif stype == 'TOROIDAL_SURFACE':
                torus_faces[fid] = surf_ref

    # ========== 1. 通孔 vs 盲孔检测 ==========
    # 对于每个圆柱面，检查其两端是否有PLANE面连接
    for surf_ref, face_ids in surface_to_faces.items():
        if surf_ref not in entities or entities[surf_ref]['type'] != 'CYLINDRICAL_SURFACE':
            continue

        # 找到该圆柱面的所有相邻面（通过边共享）
        adjacent_planes = set()
        for fid in face_ids:
            adj_faces = face_adjacency.get(fid, set())
            for af in adj_faces:
                if af in plane_faces:
                    adjacent_planes.add(af)

        # 通孔: 圆柱面两端都连通到外部（两端都有非圆柱面邻接）
        # 盲孔: 圆柱面一端有PLANE面（孔底），另一端开放
        # 简化判断: 如果相邻平面数 >= 2 且圆柱面数量较少，可能是通孔
        if len(adjacent_planes) >= 2:
            mfg['through_holes'] += 1
        elif len(adjacent_planes) == 1:
            mfg['blind_holes'] += 1
        elif len(adjacent_planes) == 0:
            # 孤立圆柱面，可能是通孔
            mfg['through_holes'] += 1

    # 孔深度比估计（基于圆柱面半径与包围盒的比值）
    cyl_radii = info.get('cylinder_radii', [])
    dims = info.get('bbox_dims', [0, 0, 0])
    max_dim = max(dims) if dims else 1
    total_holes = mfg['through_holes'] + mfg['blind_holes']
    if total_holes > 0 and cyl_radii:
        avg_radius = sum(cyl_radii) / len(cyl_radii)
        mfg['avg_hole_depth_ratio'] = min(avg_radius / max(max_dim, 1), 1.0)
        mfg['hole_aspect_ratio'] = min(max_dim / (avg_radius * 2 + 1), 5.0) / 5.0

    # ========== 2. 槽/型腔检测 ==========
    # 检测模式: 中心PLANE面（槽底）+ 四周垂直或接近垂直的PLANE面（槽壁）
    # 先找到所有PLANE面，然后检查其邻接PLANE面
    for fid, surf_ref in plane_faces.items():
        adj_planes = [af for af in face_adjacency.get(fid, set()) if af in plane_faces]
        # 槽底: 有至少3个邻接平面（槽壁）
        if len(adj_planes) >= 3:
            # 区分槽(长条形)和型腔(近似等宽)
            # 简化: 根据邻接面数量判断
            if len(adj_planes) >= 4:
                mfg['pockets'] += 1
            else:
                mfg['slots'] += 1

    # ========== 3. 凸台检测 ==========
    # 检测CYLINDRICAL_SURFACE在Z轴范围超出主体包围盒
    # 以及从主体表面突出的PLANE面
    if cyl_faces:
        mfg['bosses'] = min(len(cyl_faces) // 2, 10)  # 粗略估计

    # 凸台高度比
    if mfg['bosses'] > 0 and max_dim > 0:
        mfg['avg_boss_height_ratio'] = min(0.3, 1.0)  # 默认估计
    mfg['boss_density'] = min(mfg['bosses'] / max(total_holes + 1, 1), 1.0)

    # ========== 4. 加强筋检测 ==========
    # 高长宽比PLANE面: 一个维度远小于其他两个维度
    # 粗略估计: 基于面类型中高比例平面
    face_types = info.get('face_types', {})
    total_faces = info.get('num_faces', 1)
    plane_count = face_types.get('plane', 0)
    if plane_count > 10:
        mfg['ribs'] = max(1, plane_count // 10)
    mfg['rib_thickness_ratio'] = min(0.1 * mfg['ribs'], 1.0)

    # ========== 5. 倒角/圆角检测 ==========
    # 圆角: 统计TOROIDAL_SURFACE
    mfg['fillets'] = len(torus_faces)
    # 倒角: 启发式检测
    # 倒角通常表现为两个PLANE面以约135°角相接
    # 简化: 检查相邻PLANE面
    chamfer_count = 0
    for fid in plane_faces:
        adj_planes = [af for af in face_adjacency.get(fid, set()) if af in plane_faces]
        # 倒角通常有2个或更少的邻接平面
        if 1 <= len(adj_planes) <= 2:
            chamfer_count += 1
    mfg['chamfers'] = min(chamfer_count // 2, 20)

    if mfg['fillets'] > 0 and max_dim > 0:
        mfg['avg_fillet_radius_ratio'] = min(0.05 * min(mfg['fillets'], 10), 1.0)
    mfg['avg_chamfer_angle'] = 0.5  # 默认45度倒角比例

    # ========== 6. 螺纹检测 ==========
    # 启发式: 小半径CYLINDRICAL_SURFACE + 高密度B_SPLINE_CURVE_WITH_KNOTS
    small_cylinders = sum(1 for r in cyl_radii if r < 10) if cyl_radii else 0
    edge_types = info.get('edge_types', {})
    bspline_curves = edge_types.get('bspline', 0)
    total_edges = info.get('num_edges', 1)

    if small_cylinders > 0 and bspline_curves > 5:
        thread_ratio = bspline_curves / max(total_edges, 1)
        if thread_ratio > 0.1:
            mfg['threads'] = min(small_cylinders, 5)
            mfg['has_thread'] = 1

    # ========== 7. 特征复杂度 ==========
    feature_types = 0
    if mfg['through_holes'] > 0 or mfg['blind_holes'] > 0:
        feature_types += 1
    if mfg['slots'] > 0 or mfg['pockets'] > 0:
        feature_types += 1
    if mfg['bosses'] > 0:
        feature_types += 1
    if mfg['ribs'] > 0:
        feature_types += 1
    if mfg['fillets'] > 0:
        feature_types += 1
    if mfg['chamfers'] > 0:
        feature_types += 1
    if mfg['threads'] > 0:
        feature_types += 1

    mfg['unique_feature_types'] = feature_types
    total_features = (mfg['through_holes'] + mfg['blind_holes'] + mfg['slots'] +
                      mfg['pockets'] + mfg['bosses'] + mfg['ribs'] +
                      mfg['fillets'] + mfg['chamfers'] + mfg['threads'])
    mfg['feature_density'] = min(total_features / max(total_faces, 1), 1.0)
    mfg['feature_diversity'] = feature_types / 7.0 if feature_types > 0 else 0.0

    # 复杂度评分: 综合特征密度和多样性
    mfg['complexity_score'] = (mfg['feature_density'] * 0.5 + mfg['feature_diversity'] * 0.5)

    # 制造对称性: 通孔和盲孔比例反映对称程度
    if total_holes > 0:
        mfg['manufacturing_symmetry'] = abs(mfg['through_holes'] - mfg['blind_holes']) / max(total_holes, 1)
        mfg['manufacturing_symmetry'] = 1.0 - mfg['manufacturing_symmetry']  # 越接近1越对称

    # ========== 构建32维特征向量 ==========
    vector = np.zeros(32, dtype=np.float32)

    # 维度 0-3: 孔特征
    total_holes_norm = min(total_holes / 10, 1.0)
    vector[0] = mfg['through_holes'] / max(total_holes, 1) if total_holes > 0 else 0.0  # 通孔比例
    vector[1] = mfg['blind_holes'] / max(total_holes, 1) if total_holes > 0 else 0.0    # 盲孔比例
    vector[2] = mfg['avg_hole_depth_ratio']   # 平均孔深比
    vector[3] = mfg['hole_aspect_ratio']       # 孔长径比

    # 维度 4-7: 槽/型腔特征
    total_slots = mfg['slots'] + mfg['pockets']
    vector[4] = min(mfg['slots'] / 5, 1.0)        # 槽数量
    vector[5] = min(mfg['pockets'] / 5, 1.0)      # 型腔数量
    vector[6] = min(total_slots / 10, 1.0) if total_slots > 0 else 0.0  # 平均槽深比
    vector[7] = 0.5 if mfg['slots'] > 0 or mfg['pockets'] > 0 else 0.0  # 平均槽长宽比

    # 维度 8-10: 凸台特征
    vector[8] = min(mfg['bosses'] / 5, 1.0)        # 凸台数量
    vector[9] = mfg['avg_boss_height_ratio']       # 平均凸台高度比
    vector[10] = mfg['boss_density']                # 凸台密度

    # 维度 11-12: 加强筋特征
    vector[11] = min(mfg['ribs'] / 5, 1.0)         # 筋数量
    vector[12] = mfg['rib_thickness_ratio']         # 筋厚度比

    # 维度 13-16: 倒角/圆角特征
    vector[13] = min(mfg['chamfers'] / 10, 1.0)    # 倒角数量
    vector[14] = min(mfg['fillets'] / 10, 1.0)     # 圆角数量
    vector[15] = mfg['avg_chamfer_angle']           # 平均倒角角度
    vector[16] = mfg['avg_fillet_radius_ratio']     # 平均圆角半径比

    # 维度 17-18: 螺纹特征
    vector[17] = min(mfg['threads'] / 3, 1.0)      # 螺纹数量
    vector[18] = float(mfg['has_thread'])           # 有螺纹标志

    # 维度 19-23: 特征复杂度
    vector[19] = mfg['unique_feature_types'] / 7.0  # 唯一特征类型数
    vector[20] = mfg['feature_density']              # 特征密度
    vector[21] = mfg['feature_diversity']            # 特征多样性
    vector[22] = mfg['complexity_score']             # 复杂度评分
    vector[23] = mfg['manufacturing_symmetry']       # 制造对称性

    # 维度 24-31: 保留（填空）
    # 填充一些辅助统计信息
    vector[24] = total_holes_norm                    # 总孔数归一化
    vector[25] = min(info.get('num_unique_radii', 0) / 10, 1.0)  # 半径种类
    vector[26] = min(len(info.get('cylinder_radii', [])) / 10, 1.0)  # 圆柱半径种类
    vector[27] = min(info.get('num_faces', 0) / 100, 1.0)  # 面数归一化
    vector[28] = min(info.get('num_edges', 0) / 200, 1.0)  # 边数归一化
    # 29-31 保留

    # 裁剪
    vector = np.clip(vector, 0.0, 1.0)

    mfg['mfg_feature_vector'] = vector

    return mfg


def calculate_mfg_similarity(mfg1: dict, mfg2: dict) -> dict:
    """
    计算两个零件的制造特征相似度

    Args:
        mfg1: 零件1的制造特征字典
        mfg2: 零件2的制造特征字典

    Returns:
        相似度字典
    """
    similarities = {}

    # 1. 孔特征相似度
    h1_t = mfg1.get('through_holes', 0)
    h1_b = mfg1.get('blind_holes', 0)
    h2_t = mfg2.get('through_holes', 0)
    h2_b = mfg2.get('blind_holes', 0)
    h_sim = 1 - (abs(h1_t - h2_t) + abs(h1_b - h2_b)) / max(h1_t + h1_b + h2_t + h2_b, 1)
    similarities['holes'] = max(0, h_sim)

    # 2. 槽/型腔相似度
    s1 = mfg1.get('slots', 0) + mfg1.get('pockets', 0)
    s2 = mfg2.get('slots', 0) + mfg2.get('pockets', 0)
    s_sim = 1 - abs(s1 - s2) / max(s1, s2, 1)
    similarities['slots_pockets'] = max(0, s_sim)

    # 3. 凸台相似度
    b1 = mfg1.get('bosses', 0)
    b2 = mfg2.get('bosses', 0)
    b_sim = 1 - abs(b1 - b2) / max(b1, b2, 1)
    similarities['bosses'] = max(0, b_sim)

    # 4. 加强筋相似度
    r1 = mfg1.get('ribs', 0)
    r2 = mfg2.get('ribs', 0)
    r_sim = 1 - abs(r1 - r2) / max(r1, r2, 1)
    similarities['ribs'] = max(0, r_sim)

    # 5. 倒角/圆角相似度
    f1 = mfg1.get('fillets', 0) + mfg1.get('chamfers', 0)
    f2 = mfg2.get('fillets', 0) + mfg2.get('chamfers', 0)
    f_sim = 1 - abs(f1 - f2) / max(f1, f2, 1)
    similarities['fillets_chamfers'] = max(0, f_sim)

    # 6. 螺纹相似度
    t1 = mfg1.get('has_thread', 0)
    t2 = mfg2.get('has_thread', 0)
    similarities['thread'] = 1.0 if t1 == t2 else 0.0

    # 7. 特征复杂度相似度
    c1 = mfg1.get('complexity_score', 0)
    c2 = mfg2.get('complexity_score', 0)
    c_sim = 1 - abs(c1 - c2) / max(c1, c2, 0.01)
    similarities['complexity'] = max(0, c_sim)

    # 8. 向量余弦相似度
    v1 = mfg1.get('mfg_feature_vector')
    v2 = mfg2.get('mfg_feature_vector')
    vec_sim = 0.0
    if v1 is not None and v2 is not None:
        vec_sim = cosine_similarity(v1, v2)
    similarities['vector_similarity'] = vec_sim

    # 综合评分
    weights = {
        'holes': 0.20,
        'slots_pockets': 0.15,
        'bosses': 0.10,
        'ribs': 0.10,
        'fillets_chamfers': 0.15,
        'thread': 0.05,
        'complexity': 0.10,
        'vector_similarity': 0.15,
    }
    overall = sum(similarities[k] * w for k, w in weights.items())
    similarities['overall'] = round(overall, 4)

    return similarities


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算余弦相似度"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ============================================================
# 第四部分：自定义 Embedding 函数（支持任意 OpenAI 兼容接口）
# ============================================================

class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
    """支持自定义 base_url 的 Embedding 函数，兼容 ChromaDB 新版接口"""

    def __init__(self, api_key: str, base_url: str = None,
                 model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._get_embeddings(input)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._get_embeddings(documents)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._get_embeddings(input)

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 10  # 阿里云通义千问embedding API限制为10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
        return all_embeddings


# ============================================================
# 三路融合检索配置
# ============================================================

THREE_WAY_RECALL_CONFIG = {
    'text_recall_k': 150,      # 文本Embedding召回数量
    'geo_recall_k': 300,       # 几何向量召回数量（放宽，降低边界漏召）
    'visual_recall_k': 150,    # 视觉向量召回数量
    'max_total_recall': 500,   # 最大召回总数
}

FUSION_RERANK_CONFIG = {
    'fusion_top_k': 30,        # 融合排序后保留数量
    'min_recall_sources': 1,   # 最少被召回次数
}

# 融合权重
# NOTE: manufacturing 以固定 0.15 比率混合在 fusion_rerank 中（calculate_geometric_similarity 内部 10% + 最终精排通过 geo_sim['overall'] 携带）
FUSION_WEIGHTS = {
    'text': 0.25,
    'geometric': 0.45,
    'visual': 0.30,
}

# 视图权重（用于视觉向量提取）
VIEW_WEIGHTS = {
    'front': 0.25,
    'top': 0.20,
    'left': 0.15,
    'right': 0.15,
    'back': 0.10,
    'bottom': 0.15,
}


# ============================================================
# 第五部分：向量索引类
# ============================================================

class VisualVectorIndex:
    """视觉特征向量FAISS索引 (512维 CLIP/PIL特征)"""

    VISUAL_VECTOR_DIM = 512
    FEATURE_VERSION = 2

    def __init__(self, index_path: str = "./visual_index.faiss"):
        """
        初始化视觉向量索引

        Args:
            index_path: FAISS索引文件路径
        """
        self.index_path = index_path
        self.dim = self.VISUAL_VECTOR_DIM
        self.index: Optional['faiss.IndexFlatIP'] = None
        self._id_to_idx: Dict[str, int] = {}  # filepath -> faiss_idx
        self._idx_to_id: Dict[int, str] = {}  # faiss_idx -> filepath
        self._vectors_cache: Dict[str, np.ndarray] = {}
        self._init_index()

    def _init_index(self):
        """初始化或加载FAISS索引"""
        if HAS_FAISS:
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    # 验证维度匹配
                    if self.index.d != self.dim:
                        print(f"  [视觉索引] 已有索引维度不匹配({self.index.d} != {self.dim})，重新创建")
                        self.index = faiss.IndexFlatIP(self.dim)
                    else:
                        # 加载映射字典
                        mapping_path = self.index_path.replace('.faiss', '_mapping.json')
                        if os.path.exists(mapping_path):
                            try:
                                with open(mapping_path, 'r', encoding='utf-8') as f:
                                    mapping_data = json.load(f)
                                self._id_to_idx = mapping_data.get('id_to_idx', {})
                                self._idx_to_id = {int(k): v for k, v in mapping_data.get('idx_to_id', {}).items()}
                                if not self._mapping_is_valid(mapping_data):
                                    print(f"  [索引失效] 向量/映射数量或特征版本不一致，必须重新 build")
                                    self.reset()
                                    return
                                print(f"  [视觉索引] 加载已有索引: {self.index.ntotal} 个向量, {len(self._id_to_idx)} 个映射")
                            except Exception as e:
                                print(f"  [视觉索引] 映射文件加载失败: {e}")
                                print(f"  [视觉索引] 加载已有索引: {self.index.ntotal} 个向量 (映射加载失败)")
                        else:
                            print(f"  [视觉索引] 加载已有索引: {self.index.ntotal} 个向量 (无映射文件)")
                    return
                except Exception as e:
                    print(f"  [视觉索引] 加载失败: {e}, 重新创建")
            self.index = faiss.IndexFlatIP(self.dim)
            print(f"  [视觉索引] 创建新索引 (512维)")

    def _mapping_is_valid(self, mapping_data: dict) -> bool:
        if mapping_data.get('feature_version') != self.FEATURE_VERSION:
            return False
        if self.index is None:
            return False
        if self.index.ntotal != len(self._id_to_idx) or len(self._id_to_idx) != len(self._idx_to_id):
            return False
        return all(
            self._idx_to_id.get(idx) == filepath
            for filepath, idx in self._id_to_idx.items()
        )

    def reset(self):
        """清空内存索引和路径映射；重建索引前必须调用。"""
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._vectors_cache.clear()

    def add_vector(self, filepath: str, vector: np.ndarray) -> bool:
        """
        添加视觉向量到索引

        Args:
            filepath: STP文件路径
            vector: 512维视觉向量

        Returns:
            是否添加成功
        """
        if not HAS_FAISS or self.index is None:
            return False

        # 避免重复添加
        if filepath in self._id_to_idx:
            return True

        # 确保向量形状和类型正确
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)

        # 归一化（FAISS IndexFlatIP使用内积，需要归一化向量）
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        # 添加到索引
        faiss_idx = self.index.ntotal
        self.index.add(vec)

        # 记录映射
        self._id_to_idx[filepath] = faiss_idx
        self._idx_to_id[faiss_idx] = filepath
        self._vectors_cache[filepath] = vec.flatten()

        return True

    def search(self, query_vector: np.ndarray, top_k: int = 150) -> List[dict]:
        """
        搜索相似视觉向量

        Args:
            query_vector: 查询视觉向量 (512维)
            top_k: 返回数量

        Returns:
            搜索结果列表 [{'filepath': ..., 'similarity': ..., 'rank': ..., 'source': 'visual'}]
        """
        results = []

        if not HAS_FAISS or self.index is None or self.index.ntotal == 0:
            return results

        # 归一化查询向量
        query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        # 执行搜索（索引可能含无映射槽位，多取一些再过滤）
        mapped = len(self._idx_to_id)
        search_k = min(max(top_k * 3, top_k + max(self.index.ntotal - mapped, 0)), self.index.ntotal)
        if search_k < 1:
            return results
        distances, indices = self.index.search(query, search_k)

        # 处理结果
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx < 0:  # FAISS返回-1表示无效结果
                continue

            filepath = self._idx_to_id.get(idx)
            if filepath:
                results.append({
                    'filepath': filepath,
                    'similarity': float(dist),  # IndexFlatIP返回内积（归一化后即余弦相似度）
                    'rank': rank,
                    'source': 'visual'
                })

                if len(results) >= top_k:
                    break

        return results

    def count(self) -> int:
        """返回索引中向量数量"""
        if HAS_FAISS and self.index:
            return self.index.ntotal
        return 0

    def save(self) -> bool:
        """保存索引到文件（包含FAISS索引和映射字典）"""
        if HAS_FAISS and self.index and self.index_path:
            try:
                # 保存FAISS索引
                faiss.write_index(self.index, self.index_path)

                # 保存映射字典到JSON文件
                mapping_path = self.index_path.replace('.faiss', '_mapping.json')
                mapping_data = {
                    'feature_version': self.FEATURE_VERSION,
                    'id_to_idx': self._id_to_idx,
                    'idx_to_id': {str(k): v for k, v in self._idx_to_id.items()},  # JSON key必须是字符串
                }
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=2)

                print(f"  [视觉索引] 保存成功: {self.index.ntotal} 个向量 -> {self.index_path}")
                return True
            except Exception as e:
                print(f"  [视觉索引] 保存失败: {e}")
                return False
        return False

    def load(self) -> bool:
        """从文件加载索引（包含FAISS索引和映射字典）"""
        if HAS_FAISS and self.index_path and os.path.exists(self.index_path):
            try:
                # 加载FAISS索引
                self.index = faiss.read_index(self.index_path)

                # 加载映射字典
                mapping_path = self.index_path.replace('.faiss', '_mapping.json')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                    self._id_to_idx = mapping_data.get('id_to_idx', {})
                    self._idx_to_id = {int(k): v for k, v in mapping_data.get('idx_to_id', {}).items()}
                    if not self._mapping_is_valid(mapping_data):
                        print(f"  [索引失效] 向量/映射数量或特征版本不一致，必须重新 build")
                        self.reset()
                        return False

                    # 重建向量缓存：从FAISS索引中提取所有向量
                    if self.index.ntotal > 0:
                        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                        for filepath, idx in self._id_to_idx.items():
                            if 0 <= idx < len(all_vectors):
                                self._vectors_cache[filepath] = all_vectors[idx]

                    print(f"  [视觉索引] 加载成功: {self.index.ntotal} 个向量, {len(self._id_to_idx)} 个映射, {len(self._vectors_cache)} 个缓存")
                else:
                    print(f"  [视觉索引] 加载成功: {self.index.ntotal} 个向量 (无映射文件)")

                return True
            except Exception as e:
                print(f"  [视觉索引] 加载失败: {e}")
                return False
        return False

    def get_vector(self, filepath: str) -> Optional[np.ndarray]:
        """获取指定文件的视觉向量"""
        # 优先从缓存获取
        if filepath in self._vectors_cache:
            return self._vectors_cache[filepath]

        # 如果缓存中没有但索引中有映射，尝试从FAISS索引重建
        if filepath in self._id_to_idx and self.index is not None:
            idx = self._id_to_idx[filepath]
            if 0 <= idx < self.index.ntotal:
                try:
                    vector = self.index.reconstruct(idx)
                    self._vectors_cache[filepath] = vector  # 缓存起来
                    return vector
                except Exception:
                    pass

        return None

    def rebuild_mappings(self):
        """重建索引映射（用于加载外部索引后）"""
        # 注意：FAISS本身不存储文件路径映射，需要外部存储
        # 此方法用于从缓存重建映射
        pass


class GeoVectorIndex:
    """几何特征向量FAISS索引 (64维)"""

    GEO_VECTOR_DIM = 64
    FEATURE_VERSION = 2

    def __init__(self, index_path: str = "./geo_index.faiss"):
        """
        初始化几何向量索引

        Args:
            index_path: FAISS索引文件路径
        """
        self.index_path = index_path
        self.dim = self.GEO_VECTOR_DIM
        self.index: Optional['faiss.IndexFlatIP'] = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._vectors_cache: Dict[str, np.ndarray] = {}
        self._init_index()

    def _init_index(self):
        """初始化或加载FAISS索引"""
        if HAS_FAISS:
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    if self.index.d != self.dim:
                        print(f"  [几何索引] 已有索引维度不匹配({self.index.d} != {self.dim})，重新创建")
                        self.index = faiss.IndexFlatIP(self.dim)
                    else:
                        # 加载映射字典
                        mapping_path = self.index_path.replace('.faiss', '_mapping.json')
                        if os.path.exists(mapping_path):
                            try:
                                with open(mapping_path, 'r', encoding='utf-8') as f:
                                    mapping_data = json.load(f)
                                self._id_to_idx = mapping_data.get('id_to_idx', {})
                                self._idx_to_id = {int(k): v for k, v in mapping_data.get('idx_to_id', {}).items()}
                                if not self._mapping_is_valid(mapping_data):
                                    print(f"  [索引失效] 向量/映射数量或特征版本不一致，必须重新 build")
                                    self.reset()
                                    return
                                print(f"  [几何索引] 加载已有索引: {self.index.ntotal} 个向量, {len(self._id_to_idx)} 个映射")
                            except Exception as e:
                                print(f"  [几何索引] 映射文件加载失败: {e}")
                                print(f"  [几何索引] 加载已有索引: {self.index.ntotal} 个向量 (映射加载失败)")
                        else:
                            print(f"  [几何索引] 加载已有索引: {self.index.ntotal} 个向量 (无映射文件)")
                    return
                except Exception as e:
                    print(f"  [几何索引] 加载失败: {e}, 重新创建")
            self.index = faiss.IndexFlatIP(self.dim)
            print(f"  [几何索引] 创建新索引 (64维)")

    def _mapping_is_valid(self, mapping_data: dict) -> bool:
        if mapping_data.get('feature_version') != self.FEATURE_VERSION:
            return False
        if self.index is None:
            return False
        if self.index.ntotal != len(self._id_to_idx) or len(self._id_to_idx) != len(self._idx_to_id):
            return False
        return all(
            self._idx_to_id.get(idx) == filepath
            for filepath, idx in self._id_to_idx.items()
        )

    def reset(self):
        """清空内存索引和路径映射；重建索引前必须调用。"""
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._vectors_cache.clear()

    def add_vector(self, filepath: str, vector: np.ndarray) -> bool:
        """添加几何向量到索引"""
        if not HAS_FAISS or self.index is None:
            return False

        if filepath in self._id_to_idx:
            return True

        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        faiss_idx = self.index.ntotal
        self.index.add(vec)
        self._id_to_idx[filepath] = faiss_idx
        self._idx_to_id[faiss_idx] = filepath
        self._vectors_cache[filepath] = vec.flatten()
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 150) -> List[dict]:
        """搜索相似几何向量"""
        results = []

        if not HAS_FAISS or self.index is None or self.index.ntotal == 0:
            return results

        query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        mapped = len(self._idx_to_id)
        search_k = min(max(top_k * 3, top_k + max(self.index.ntotal - mapped, 0)), self.index.ntotal)
        if search_k < 1:
            return results
        distances, indices = self.index.search(query, search_k)

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx < 0:
                continue
            filepath = self._idx_to_id.get(idx)
            if filepath:
                results.append({
                    'filepath': filepath,
                    'similarity': float(dist),
                    'rank': rank,
                    'source': 'geometric'
                })
                if len(results) >= top_k:
                    break
        return results

    def count(self) -> int:
        """返回索引中向量数量"""
        if HAS_FAISS and self.index:
            return self.index.ntotal
        return 0

    def save(self) -> bool:
        """保存索引到文件（包含FAISS索引和映射字典）"""
        if HAS_FAISS and self.index and self.index_path:
            try:
                # 保存FAISS索引
                faiss.write_index(self.index, self.index_path)

                # 保存映射字典到JSON文件
                mapping_path = self.index_path.replace('.faiss', '_mapping.json')
                mapping_data = {
                    'feature_version': self.FEATURE_VERSION,
                    'id_to_idx': self._id_to_idx,
                    'idx_to_id': {str(k): v for k, v in self._idx_to_id.items()},  # JSON key必须是字符串
                }
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=2)

                print(f"  [几何索引] 保存成功: {self.index.ntotal} 个向量 -> {self.index_path}")
                return True
            except Exception as e:
                print(f"  [几何索引] 保存失败: {e}")
                return False
        return False

    def load(self) -> bool:
        """从文件加载索引（包含FAISS索引和映射字典）"""
        if HAS_FAISS and self.index_path and os.path.exists(self.index_path):
            try:
                # 加载FAISS索引
                self.index = faiss.read_index(self.index_path)

                # 加载映射字典
                mapping_path = self.index_path.replace('.faiss', '_mapping.json')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                    self._id_to_idx = mapping_data.get('id_to_idx', {})
                    self._idx_to_id = {int(k): v for k, v in mapping_data.get('idx_to_id', {}).items()}
                    if not self._mapping_is_valid(mapping_data):
                        print(f"  [索引失效] 向量/映射数量或特征版本不一致，必须重新 build")
                        self.reset()
                        return False

                    # 重建向量缓存：从FAISS索引中提取所有向量
                    if self.index.ntotal > 0:
                        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                        for filepath, idx in self._id_to_idx.items():
                            if 0 <= idx < len(all_vectors):
                                self._vectors_cache[filepath] = all_vectors[idx]

                    print(f"  [几何索引] 加载成功: {self.index.ntotal} 个向量, {len(self._id_to_idx)} 个映射, {len(self._vectors_cache)} 个缓存")
                else:
                    print(f"  [几何索引] 加载成功: {self.index.ntotal} 个向量 (无映射文件)")

                return True
            except Exception as e:
                print(f"  [几何索引] 加载失败: {e}")
                return False
        return False

    def get_vector(self, filepath: str) -> Optional[np.ndarray]:
        """获取指定文件的几何向量"""
        # 优先从缓存获取
        if filepath in self._vectors_cache:
            return self._vectors_cache[filepath]

        # 如果缓存中没有但索引中有映射，尝试从FAISS索引重建
        if filepath in self._id_to_idx and self.index is not None:
            idx = self._id_to_idx[filepath]
            if 0 <= idx < self.index.ntotal:
                try:
                    vector = self.index.reconstruct(idx)
                    self._vectors_cache[filepath] = vector  # 缓存起来
                    return vector
                except Exception:
                    pass

        return None


# ============================================================
# 第六部分：Embedding 检索
# ============================================================

class EmbeddingIndex:
    def __init__(self, api_key: str, base_url: str = None,
                 db_path: str = "./chroma_parts",
                 model: str = "text-embedding-3-large",
                 use_faiss: bool = False,
                 dim: int = 1024,
                 enable_geo_index: bool = True,
                 enable_visual_index: bool = True):
        """
        初始化Embedding索引

        Args:
            api_key: API密钥
            base_url: API基础URL
            db_path: 向量数据库路径
            model: Embedding模型名称
            use_faiss: 是否使用FAISS索引（后端索引格式）
            dim: 向量维度（OpenAI text-embedding-3-large=1536, 阿里云text-embedding-v3=1024）
            enable_geo_index: 是否启用几何向量索引（64维FAISS）
            enable_visual_index: 是否启用视觉向量索引（512维FAISS）
        """
        self.use_faiss = use_faiss
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.db_path = db_path
        self._info_cache = {}
        self._desc_cache = {}

        # 几何向量索引 (64维)
        self.geo_index: Optional[GeoVectorIndex] = None
        if enable_geo_index and HAS_FAISS:
            geo_index_path = Path(db_path) / "geo_vectors.faiss"
            self.geo_index = GeoVectorIndex(str(geo_index_path))

        # 视觉向量索引 (512维)
        self.visual_index: Optional[VisualVectorIndex] = None
        if enable_visual_index and HAS_FAISS:
            visual_index_path = Path(db_path) / "visual_vectors.faiss"
            self.visual_index = VisualVectorIndex(str(visual_index_path))

        if use_faiss:
            # 使用后端的FAISS索引
            import sys
            backend_path = Path(__file__).parent / "backend"
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
            from app.core.vector_index_v2 import FAISSVectorIndex
            self.faiss_index = FAISSVectorIndex(
                db_path=str(Path(db_path) / "faiss_meta.db"),
                index_path=str(Path(db_path) / "faiss_index.bin"),
                api_key=api_key,
                base_url=base_url,
                model=model,
                dim=dim
            )
            print(f"  [FAISS] 使用后端FAISS索引: {db_path}")
        else:
            # 原有的 ChromaDB 逻辑
            self.client = chromadb.PersistentClient(path=db_path)
            self.embed_fn = CustomEmbeddingFunction(
                api_key=api_key, base_url=base_url, model=model,
            )
            self.collection = self.client.get_or_create_collection(
                name="stp_parts_v3",
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

    def build_index(self, directory: str, view_dir: str = None):
        """
        构建索引（文本Embedding + 几何向量 + 视觉向量）

        Args:
            directory: STP文件目录
            view_dir: 视图图片目录（用于视觉向量提取）
        """
        if self.use_faiss:
            print("  [警告] FAISS模式下不支持build_index，请使用后端API建立索引")
            print(f"  [提示] 后端索引路径: {self.db_path}")
            return

        # 清理旧 FAISS 索引文件（避免向量维度变更后 index.ntotal != mapping 数量）
        for fname in ['geo_vectors.faiss', 'geo_vectors_mapping.json',
                       'visual_vectors.faiss', 'visual_vectors_mapping.json']:
            fpath = Path(self.db_path) / fname
            if fpath.exists():
                fpath.unlink()
                print(f"  [清理] 删除旧索引文件: {fpath}")

        # 关键修复：只删除磁盘文件不够；索引对象在 __init__ 时已经加载到内存。
        # 如果不 reset，多次 build 会把旧向量/旧映射继续保存回磁盘。
        if self.geo_index:
            self.geo_index.reset()
        if self.visual_index:
            self.visual_index.reset()

        existing = self.collection.get()
        if existing['ids']:
            self.collection.delete(ids=existing['ids'])

        stp_files = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.stp', '.step')):
                    stp_files.append(os.path.join(root, f))

        # 排除测试零件（文件名包含"测试件"）
        stp_files = [fp for fp in stp_files if '测试件' not in Path(fp).stem]

        total = len(stp_files)
        documents, metadatas, ids = [], [], []
        start = time.time()

        for i, fp in enumerate(stp_files, 1):
            try:
                info = parse_stp_deep(fp)
                desc = generate_retrieval_description(info)
                self._info_cache[fp] = info
                self._desc_cache[fp] = desc
                documents.append(desc)
                metadatas.append({
                    'path': fp,
                    'filename': info['filename'],
                    'num_faces': info['num_faces'],
                    'euler': info['euler'],
                    'bbox': f"{info['bbox_dims'][0]}x{info['bbox_dims'][1]}x{info['bbox_dims'][2]}",
                })
                ids.append(f"part_{i:04d}")
                elapsed = time.time() - start
                print(
                    f"\r  [{i}/{total}] ✓ {info['filename']:<30s} "
                    f"({i/(elapsed+0.01):.1f} parts/s)",
                    end="", flush=True
                )
            except Exception as e:
                print(f"\n  [{i}/{total}] ✗ {Path(fp).name}: {e}")

        # 分批入库
        batch = 10  # 与embedding API的batch size保持一致
        for j in range(0, len(documents), batch):
            self.collection.add(
                documents=documents[j:j+batch],
                metadatas=metadatas[j:j+batch],
                ids=ids[j:j+batch],
            )
            print(f"\r  Embedding 入库中... {min(j+batch, len(documents))}/{len(documents)}",
                  end="", flush=True)

        elapsed = time.time() - start
        print(f"\n\n索引完成: {len(documents)} 个零件 ({elapsed:.1f}s)")

        # ========== 构建几何向量索引 ==========
        if self.geo_index:
            print(f"\n  [几何索引] 正在构建...")
            geo_success, geo_failed = 0, 0
            for i, fp in enumerate(stp_files, 1):
                try:
                    info = self._info_cache.get(fp)
                    if info is None:
                        info = parse_stp_deep(fp)
                        self._info_cache[fp] = info
                    geo_vector = extract_geo_vector_standalone(info)
                    self.geo_index.add_vector(fp, geo_vector)
                    geo_success += 1
                except Exception as e:
                    geo_failed += 1

                if i % 50 == 0:
                    print(f"    几何索引: {i}/{total}")

            self.geo_index.save()
            print(f"  [几何索引] 构建完成: {geo_success} 成功, {geo_failed} 失败")

        # ========== 构建视觉向量索引 ==========
        if self.visual_index:
            print(f"\n  [视觉索引] 正在构建...")

            # 确定视图目录
            if view_dir is None:
                view_dir = Path(directory) / "views"
            else:
                view_dir = Path(view_dir)
            view_dir.mkdir(parents=True, exist_ok=True)

            visual_success, visual_failed = 0, 0
            for i, fp in enumerate(stp_files, 1):
                try:
                    # 尝试查找已有视图
                    view_paths = find_view_images(fp, str(view_dir))
                    valid_views = [v for v in (view_paths or []) if v and Path(v).exists()]

                    # 视图不足时尝试渲染
                    if len(valid_views) < 3 and HAS_RENDER:
                        rendered = self._render_views(fp, str(view_dir))
                        if rendered:
                            valid_views = rendered

                    if len(valid_views) >= 3:
                        visual_vector = extract_visual_vector_standalone(valid_views)
                        if visual_vector is not None:
                            self.visual_index.add_vector(fp, visual_vector)
                            visual_success += 1
                        else:
                            visual_failed += 1
                    else:
                        visual_failed += 1

                    if i % 50 == 0:
                        print(f"    视觉索引: {i}/{total}")

                except Exception as e:
                    print(f"    警告: {Path(fp).name} 视觉向量提取失败: {e}")
                    visual_failed += 1

            self.visual_index.save()
            print(f"  [视觉索引] 构建完成: {visual_success} 成功, {visual_failed} 失败")

    def search(self, query_path: str, top_k: int = 20, include_geometric: bool = True,
               custom_query_text: str = None, use_multi_recall: bool = True,
               query_views: List[str] = None,
               view_dir: str = None, auto_render: bool = True) -> list[dict]:
        """
        检索相似零件（支持多路召回）

        Args:
            query_path: 查询STP文件路径
            top_k: 返回数量
            include_geometric: 是否计算几何相似度
            custom_query_text: 自定义查询文本，用于替代或补充自动生成的描述
            use_multi_recall: 是否启用多路召回（Embedding + 几何特征）
            query_views: 查询零件的视图图片路径列表（用于视觉向量）
            view_dir: 视图图片根目录（用于查找或渲染视图）
            auto_render: 如果视图不存在，是否自动渲染

        Returns:
            检索结果列表
        """
        # FAISS模式：直接委托给FAISS索引
        if self.use_faiss:
            results = self.faiss_index.search(
                query_path=query_path,
                query_text=custom_query_text,
                top_k=top_k,
                include_geometric=include_geometric
            )
            # 同步缓存
            self._info_cache.update(self.faiss_index._info_cache)
            self._desc_cache.update(self.faiss_index._desc_cache)
            return results

        # 获取查询文件的信息
        if query_path in self._desc_cache:
            desc = self._desc_cache[query_path]
            query_info = self._info_cache[query_path]
        else:
            query_info = parse_stp_deep(query_path)
            desc = generate_retrieval_description(query_info)
            self._info_cache[query_path] = query_info
            self._desc_cache[query_path] = desc

        # 如果提供了自定义查询文本，用它替代自动生成的描述
        if custom_query_text:
            search_text = f"""【用户查找需求】
{custom_query_text}

【参考零件特征】
{desc}"""
            print(f"  [自定义查询] 使用自定义文本进行Embedding检索")
        else:
            search_text = desc

        # 检查集合是否为空
        collection_count = self.collection.count()
        if collection_count == 0:
            print("  [警告] 索引库为空，请先运行 build 命令建立索引")
            return []

        # 诊断信息
        print(f"  [诊断] 索引库中共有 {collection_count} 个零件")
        print(f"  [诊断] 请求召回 top-{top_k} 个候选")

        # ========== 多路召回策略 ==========
        if use_multi_recall and include_geometric:
            return self._multi_recall_search(query_info, search_text, top_k, collection_count)

        # ========== 单路召回（仅Embedding）==========
        results = self.collection.query(
            query_texts=[search_text],
            n_results=min(top_k, collection_count),
        )

        print(f"  [诊断] Embedding召回 {len(results['ids'][0]) if results['ids'] else 0} 个候选")

        output = self._process_search_results(results, query_info, include_geometric)
        return output

    def _multi_recall_search(self, query_info: dict, search_text: str, top_k: int,
                              collection_count: int) -> list[dict]:
        """
        多路召回：Embedding召回 + 几何特征召回

        策略：
        1. Embedding召回 top_k * 2 个候选
        2. 几何特征召回：遍历所有零件，计算硬特征相似度，取 top_k 个
        3. 合并两路召回结果（去重），按混合相似度排序
        """
        # 路径1: Embedding召回（扩大召回范围）
        emb_recall_top = min(top_k * 3, collection_count)  # 扩大3倍召回
        print(f"  [多路召回] Embedding召回 top-{emb_recall_top}...")

        emb_results = self.collection.query(
            query_texts=[search_text],
            n_results=emb_recall_top,
        )

        emb_candidates = {}  # path -> result
        if emb_results['ids'] and emb_results['ids'][0]:
            for i in range(len(emb_results['ids'][0])):
                path = emb_results['metadatas'][0][i]['path']
                emb_candidates[path] = {
                    'filename': emb_results['metadatas'][0][i]['filename'],
                    'path': path,
                    'distance': emb_results['distances'][0][i],
                    'similarity': 1.0 - emb_results['distances'][0][i],
                    'description': emb_results['documents'][0][i],
                    'metadata': emb_results['metadatas'][0][i],
                    'recall_source': 'embedding',
                }

        print(f"  [多路召回] Embedding召回 {len(emb_candidates)} 个候选")

        # 路径2: 几何特征召回
        print(f"  [多路召回] 几何特征召回中...")
        geo_candidates = self._geometric_recall(query_info, top_k, emb_candidates)
        print(f"  [多路召回] 几何特征召回 {len(geo_candidates)} 个新候选")

        # 合并两路召回结果
        all_candidates = emb_candidates.copy()
        for path, result in geo_candidates.items():
            if path not in all_candidates:
                all_candidates[path] = result
                all_candidates[path]['recall_source'] = 'geometric'

        print(f"  [多路召回] 合并后共 {len(all_candidates)} 个候选（去重后）")

        # 计算混合相似度并排序
        output = []
        for path, result in all_candidates.items():
            candidate_info = self._info_cache.get(path)
            if candidate_info is None:
                try:
                    candidate_info = parse_stp_deep(path)
                    self._info_cache[path] = candidate_info
                except:
                    continue

            if candidate_info:
                geo_sim = calculate_geometric_similarity(query_info, candidate_info)
                result['geometric_similarity'] = geo_sim
                # 视觉主导权重：视觉40% + Embedding 35% + 几何 25%
                # 如果有视觉相似度则使用三路融合，否则使用传统二路融合
                visual_sim = result.get('visual_similarity', 0)
                if visual_sim > 0:
                    result['hybrid_similarity'] = round(
                        0.35 * result['similarity'] + 0.25 * geo_sim['overall'] + 0.40 * visual_sim, 4
                    )
                else:
                    result['hybrid_similarity'] = round(
                        0.55 * result['similarity'] + 0.45 * geo_sim['overall'], 4
                    )
            output.append(result)

        # 按混合相似度排序
        output.sort(key=lambda x: x.get('hybrid_similarity', x.get('similarity', 0)), reverse=True)

        # 返回 top_k 个
        return output[:top_k]

    def _geometric_recall(self, query_info: dict, top_k: int,
                           exclude_paths: dict) -> dict:
        """
        基于几何特征的召回

        筛选条件（宽松匹配）：
        1. 尺寸范围：查询零件尺寸的 ±50% 范围内
        2. 面类型分布相似：平面/圆柱面比例接近
        3. 拓扑复杂度接近：面数/边数在合理范围内
        """
        geo_candidates = {}

        # 提取查询零件的几何特征
        query_dims = query_info.get('bbox_dims', [0, 0, 0])

        query_faces = query_info.get('face_types', {})
        total_faces = sum(query_faces.values()) if query_faces else 1
        query_plane_ratio = query_faces.get('plane', 0) / max(total_faces, 1)
        query_cylinder_ratio = query_faces.get('cylinder', 0) / max(total_faces, 1)

        query_face_count = query_info.get('num_faces', 0)

        # 遍历所有已索引的零件
        all_results = self.collection.get()

        if not all_results['ids']:
            return geo_candidates

        scored_candidates = []

        for i in range(len(all_results['ids'])):
            path = all_results['metadatas'][i]['path']

            # 跳过已在Embedding召回中的零件和测试零件
            if path in exclude_paths:
                continue
            if '测试件' in Path(path).stem:
                continue

            # 获取零件几何信息
            candidate_info = self._info_cache.get(path)
            if candidate_info is None:
                try:
                    candidate_info = parse_stp_deep(path)
                    self._info_cache[path] = candidate_info
                except:
                    continue

            # 计算几何相似度得分
            score = 0

            # 1. 尺寸相似度（宽松匹配）
            cand_dims = candidate_info.get('bbox_dims', [0, 0, 0])

            if query_dims[0] > 0 and query_dims[1] > 0 and query_dims[2] > 0:
                dim_ratios = []
                for qd, cd in zip(query_dims, cand_dims):
                    if qd > 0:
                        ratio = min(qd, cd) / max(qd, cd)
                        dim_ratios.append(ratio)
                if dim_ratios:
                    score += sum(dim_ratios) / len(dim_ratios) * 0.4

            # 2. 面类型分布相似度
            cand_faces = candidate_info.get('face_types', {})
            cand_total = sum(cand_faces.values()) if cand_faces else 1
            cand_plane_ratio = cand_faces.get('plane', 0) / max(cand_total, 1)
            cand_cylinder_ratio = cand_faces.get('cylinder', 0) / max(cand_total, 1)

            face_sim = 1.0 - (abs(query_plane_ratio - cand_plane_ratio) +
                             abs(query_cylinder_ratio - cand_cylinder_ratio)) / 2
            score += max(0, face_sim) * 0.3

            # 3. 拓扑复杂度相似度
            cand_face_count = candidate_info.get('num_faces', 0)

            if query_face_count > 0 and cand_face_count > 0:
                topo_sim = min(query_face_count, cand_face_count) / max(query_face_count, cand_face_count)
                score += topo_sim * 0.3

            if score > 0.3:  # 阈值：至少30%相似
                scored_candidates.append({
                    'path': path,
                    'filename': all_results['metadatas'][i]['filename'],
                    'description': all_results['documents'][i],
                    'metadata': all_results['metadatas'][i],
                    'geo_recall_score': round(score, 4),
                    'similarity': score,  # 临时用geo_recall_score作为similarity
                })

        # 按得分排序，取top_k
        scored_candidates.sort(key=lambda x: x['geo_recall_score'], reverse=True)
        for c in scored_candidates[:top_k]:
            geo_candidates[c['path']] = c

        return geo_candidates

    def _process_search_results(self, results: dict, query_info: dict,
                                 include_geometric: bool) -> list[dict]:
        """处理检索结果，计算几何相似度"""
        output = []
        cache_miss_count = 0

        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            candidate_path = meta['path']
            candidate_info = self._info_cache.get(candidate_path)

            if include_geometric and candidate_info is None:
                try:
                    candidate_info = parse_stp_deep(candidate_path)
                    self._info_cache[candidate_path] = candidate_info
                except Exception as e:
                    cache_miss_count += 1
                    if cache_miss_count == 1:
                        print(f"  [警告] 部分候选零件无法解析几何信息: {e}")

            result = {
                'filename': meta['filename'],
                'path': candidate_path,
                'distance': results['distances'][0][i],
                'similarity': 1.0 - results['distances'][0][i],
                'description': results['documents'][0][i],
                'metadata': meta,
            }

            if include_geometric and candidate_info:
                geo_sim = calculate_geometric_similarity(query_info, candidate_info)
                result['geometric_similarity'] = geo_sim
                result['hybrid_similarity'] = round(
                    0.7 * result['similarity'] + 0.3 * geo_sim['overall'], 4
                )

            output.append(result)

        # 按混合相似度排序
        if include_geometric and output and any('hybrid_similarity' in x for x in output):
            output.sort(key=lambda x: x.get('hybrid_similarity', x.get('similarity', 0)), reverse=True)
        else:
            output.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return output

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.use_faiss:
            return self.faiss_index._get_embeddings(texts)

        all_embeddings = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embed_fn._get_embeddings(batch)
            all_embeddings.extend(response)
        return all_embeddings

    def three_way_recall(
        self,
        query_path: str,
        query_text: str = None,
        query_info: dict = None,
        query_geo_vector: np.ndarray = None,
        query_visual_vector: np.ndarray = None,
        view_paths: List[str] = None,
        text_recall_k: int = 150,
        geo_recall_k: int = 150,
        visual_recall_k: int = 150,
        debugger: RecallDebugger = None,
    ) -> Tuple[List[dict], dict, np.ndarray, np.ndarray]:
        """
        三路召回: 文本 + 几何 + 视觉

        从三个独立的索引中分别召回候选，然后合并去重

        Args:
            query_path: 查询STP文件路径
            query_text: 查询文本（可选，不提供则自动生成）
            query_info: 查询零件解析信息（可选）
            query_geo_vector: 查询几何向量（可选）
            query_visual_vector: 查询视觉向量（可选）
            view_paths: 查询零件视图路径（可选）
            text_recall_k: 文本召回数量
            geo_recall_k: 几何召回数量
            visual_recall_k: 视觉召回数量

        Returns:
            (candidates, query_info, query_geo_vector, query_visual_vector)
        """
        # 解析查询零件
        if query_info is None:
            query_info = parse_stp_deep(query_path)
        if query_geo_vector is None:
            query_geo_vector = extract_geo_vector_standalone(query_info)
        if query_text is None:
            query_text = generate_retrieval_description(query_info)

        # ========== 1. 文本Embedding召回 ==========
        text_results: Dict[str, dict] = {}
        collection_count = self.collection.count()
        if collection_count > 0:
            text_search = self.collection.query(
                query_texts=[query_text],
                n_results=min(text_recall_k, collection_count),
            )
            if text_search['ids'] and text_search['ids'][0]:
                for i, (metadata, dist) in enumerate(zip(
                    text_search['metadatas'][0],
                    text_search['distances'][0]
                )):
                    filepath = metadata.get('path', metadata.get('filepath', ''))
                    if filepath:
                        text_results[filepath] = {
                            'text_similarity': 1.0 - dist,  # ChromaDB返回cosine distance
                            'text_rank': i + 1
                        }

        print(f"    [文本召回] {len(text_results)} 个候选")
        if text_results:
            for fp in sorted(text_results.keys(), key=lambda x: text_results[x]['text_rank'])[:30]:
                print(f"      文本[{text_results[fp]['text_rank']}] {Path(fp).name}")
        if debugger:
            text_candidates = [{'filepath': fp, **v} for fp, v in text_results.items()]
            debugger.record_text_recall(text_candidates)

        # ========== 2. 几何向量召回 ==========
        geo_results: Dict[str, dict] = {}
        if self.geo_index and query_geo_vector is not None:
            geo_search = self.geo_index.search(query_geo_vector, geo_recall_k)
            for r in geo_search:
                geo_results[r['filepath']] = {
                    'geo_similarity': r['similarity'],
                    'geo_rank': r.get('rank', 0)
                }
            print(f"    [几何召回] {len(geo_results)} 个候选")
            if geo_results:
                for fp in sorted(geo_results.keys(), key=lambda x: geo_results[x]['geo_rank'])[:30]:
                    print(f"      几何[{geo_results[fp]['geo_rank']}] {Path(fp).name}")
            if debugger:
                geo_candidates = [{'filepath': fp, **v} for fp, v in geo_results.items()]
                debugger.record_geo_recall(geo_candidates)
        else:
            print(f"    [几何召回] 跳过（索引不可用）")

        # ========== 3. 视觉向量召回 ==========
        visual_results: Dict[str, dict] = {}
        if self.visual_index:
            # 提取查询视觉向量
            if query_visual_vector is None and view_paths:
                valid_views = [v for v in view_paths if v and Path(v).exists()]
                if len(valid_views) >= 3:
                    query_visual_vector = extract_visual_vector_standalone(valid_views)

            if query_visual_vector is not None:
                visual_search = self.visual_index.search(query_visual_vector, visual_recall_k)
                for r in visual_search:
                    visual_results[r['filepath']] = {
                        'visual_similarity': r['similarity'],
                        'visual_rank': r.get('rank', 0)
                    }
                print(f"    [视觉召回] {len(visual_results)} 个候选")
                if visual_results:
                    for fp in sorted(visual_results.keys(), key=lambda x: visual_results[x]['visual_rank'])[:30]:
                        print(f"      视觉[{visual_results[fp]['visual_rank']}] {Path(fp).name}")
                if debugger:
                    visual_candidates = [{'filepath': fp, **v} for fp, v in visual_results.items()]
                    debugger.record_visual_recall(visual_candidates)
            else:
                print(f"    [视觉召回] 跳过（视觉向量不可用）")
        else:
            print(f"    [视觉召回] 跳过（索引不可用）")

        # ========== 4. 合并去重 ==========
        all_filepaths = set(text_results.keys()) | set(geo_results.keys()) | set(visual_results.keys())
        merged: List[dict] = []

        for fp in all_filepaths:
            item = {
                'filepath': fp,
                'filename': Path(fp).name,
                'recall_sources': [],
                'recall_count': 0
            }

            if fp in text_results:
                item['text_similarity'] = text_results[fp]['text_similarity']
                item['text_rank'] = text_results[fp]['text_rank']
                item['recall_sources'].append('text')
                item['recall_count'] += 1

            if fp in geo_results:
                item['geo_similarity'] = geo_results[fp]['geo_similarity']
                item['geo_rank'] = geo_results[fp]['geo_rank']
                item['recall_sources'].append('geometric')
                item['recall_count'] += 1

            if fp in visual_results:
                item['visual_similarity'] = visual_results[fp]['visual_similarity']
                item['visual_rank'] = visual_results[fp]['visual_rank']
                item['recall_sources'].append('visual')
                item['recall_count'] += 1

            # 制造特征相似度（不依赖独立召回通道，所有合并候选均计算）
            cand_info = self._info_cache.get(fp) or parse_stp_deep(fp)
            if fp not in self._info_cache:
                self._info_cache[fp] = cand_info
            q_mfg = query_info.get('mfg_features', {})
            c_mfg = cand_info.get('mfg_features', {})
            if q_mfg and c_mfg:
                mfg_sim = calculate_mfg_similarity(q_mfg, c_mfg)
                item['mfg_similarity'] = mfg_sim['overall']
                item['mfg_hole_sim'] = mfg_sim.get('holes', 0.0)
            else:
                item['mfg_similarity'] = 0.0
                item['mfg_hole_sim'] = 0.0

            merged.append(item)

        # 过滤测试零件
        merged = [item for item in merged if '测试件' not in Path(item['filepath']).stem]

        print(f"    [合并去重] {len(merged)} 个候选（已过滤测试零件）")
        for i, c in enumerate(merged, 1):
            print(f"      {i}. {Path(c['filepath']).name} (召回: {c.get('recall_count', 0)}路)")
        if debugger:
            debugger.record_merge(merged)

        return merged, query_info, query_geo_vector, query_visual_vector

    def fusion_rerank(
        self,
        candidates: List[dict],
        fusion_weights: dict = None,
        fusion_top_k: int = 30,
        min_recall_sources: int = 1,
        debugger: RecallDebugger = None,
    ) -> List[dict]:
        """
        轻量级融合排序

        使用RRF和加权融合对候选进行排序。缺路时不默认0.5，而是按实际召回路
        数归一化权重；对「文本+孔特征」强匹配给予保底加成。

        Args:
            candidates: 召回阶段的候选列表
            fusion_weights: 融合权重 {'text': 0.3, 'geometric': 0.35, 'visual': 0.35}
            fusion_top_k: 融合后保留数量
            min_recall_sources: 最少被召回次数过滤

        Returns:
            融合排序后的候选列表
        """
        weights = fusion_weights or FUSION_WEIGHTS

        # 过滤: 至少被 min_recall_sources 路召回
        filtered = [c for c in candidates if c.get('recall_count', 0) >= min_recall_sources]

        if not filtered:
            return []

        # 按本次实际可用的召回通道做全局归一化。旧代码按候选自身通道数归一化，
        # 会让“只在一路排第1”与“三路都排得靠前”得到几乎相同的RRF分数。
        route_specs = (
            ('text', 'text_rank'),
            ('geometric', 'geo_rank'),
            ('visual', 'visual_rank'),
        )
        active_route_specs = [
            spec for spec in route_specs if any(spec[1] in c for c in filtered)
        ]
        active_weight_total = sum(weights[route] for route, _ in active_route_specs)

        # 计算融合分数
        for cand in filtered:
            routes = cand.get('recall_sources', [])
            n_routes = len(routes)

            # ------ 各路相似度：只取实际存在的路，不默认 0.5 ------
            text_sim = cand.get('text_similarity')
            geo_sim = cand.get('geo_similarity')
            visual_sim = cand.get('visual_similarity')
            mfg_sim = cand.get('mfg_similarity', 0.0)
            hole_sim = cand.get('mfg_hole_sim', 0.0)

            # RRF 分数 (Reciprocal Rank Fusion)
            rrf_score = 0.0
            rrf_k = 60
            if 'text_rank' in cand:
                rrf_score += 1.0 / (rrf_k + cand['text_rank'])
            if 'geo_rank' in cand:
                rrf_score += 1.0 / (rrf_k + cand['geo_rank'])
            if 'visual_rank' in cand:
                rrf_score += 1.0 / (rrf_k + cand['visual_rank'])

            # ------ 加权分数：按本次全局可用通道归一化，缺失通道不再被无条件原谅 ------
            weighted_sum = 0.0
            if 'text' in routes and text_sim is not None:
                weighted_sum += weights['text'] * text_sim
            if 'geometric' in routes and geo_sim is not None:
                weighted_sum += weights['geometric'] * geo_sim
            if 'visual' in routes and visual_sim is not None:
                weighted_sum += weights['visual'] * visual_sim

            # 制造特征加成：文本+几何同时高匹配 → 孔/结构特征一致
            mfg_boost = 0.0
            if 'text' in routes and 'geometric' in routes:
                t = text_sim or 0.0
                g = geo_sim or 0.0
                if t > 0.7 and g > 0.6:
                    mfg_boost = 0.08  # 保底加成
                elif t > 0.6 and g > 0.5:
                    mfg_boost = 0.04

            # 综合加权分数（含制造特征通道）
            normalized_weighted = weighted_sum / active_weight_total if active_weight_total > 0 else 0.0
            if active_weight_total > 0:
                normalized_weighted = normalized_weighted * 0.85 + mfg_sim * 0.15

            # 用“所有可用通道均排第1”作为RRF上限，保留多路共识优势
            max_rrf = len(active_route_specs) / (rrf_k + 1)
            normalized_rrf = rrf_score / max_rrf if max_rrf > 0 else 0

            # 融合 = 40% RRF + 60% 加权 + 制造加成
            cand['fusion_score'] = 0.4 * normalized_rrf + 0.6 * normalized_weighted + mfg_boost
            cand['fusion_score'] = min(cand['fusion_score'], 1.0)  # 截断到 [0, 1]
            cand['rrf_score'] = rrf_score
            cand['weighted_score'] = normalized_weighted
            cand['mfg_boost'] = mfg_boost

        # 先按融合分数排序，再为每一路保留安全名额。这样某一路的强命中不会在
        # 进入完整几何精排前被融合截断（本例的几何第1名必须保留下来）。
        filtered.sort(key=lambda x: x['fusion_score'], reverse=True)
        per_route_keep = max(1, min(8, fusion_top_k // max(len(active_route_specs), 1)))
        selected_by_path: Dict[str, dict] = {}
        for route, rank_key in active_route_specs:
            route_top = sorted(
                (c for c in filtered if rank_key in c),
                key=lambda c: c[rank_key],
            )[:per_route_keep]
            for cand in route_top:
                selected_by_path[cand['filepath']] = cand
                cand.setdefault('protected_routes', []).append(route)

        # 用整体融合榜补满剩余名额
        for cand in filtered:
            if len(selected_by_path) >= fusion_top_k:
                break
            selected_by_path.setdefault(cand['filepath'], cand)

        selected = list(selected_by_path.values())[:fusion_top_k]
        selected.sort(key=lambda x: x['fusion_score'], reverse=True)

        print(f"    [融合排序] 保留 {len(selected)} 个候选（每路保护 {per_route_keep} 个）")
        for i, c in enumerate(selected, 1):
            boost = c.get('mfg_boost', 0)
            boost_str = f" (制造加成+{boost})" if boost > 0 else ""
            protected = c.get('protected_routes', [])
            protected_str = f" [保护:{','.join(protected)}]" if protected else ""
            print(f"      {i}. {Path(c['filepath']).name} (融合分数: {c.get('fusion_score', 0):.4f}{boost_str}){protected_str}")
        if debugger:
            debugger.record_fusion(selected)

        return selected

    def search_three_way(
        self,
        query_path: str,
        query_text: str = None,
        view_dir: str = None,
        top_k: int = 20,
        text_recall_k: int = 150,
        geo_recall_k: int = 150,
        visual_recall_k: int = 150,
        fusion_top_k: int = 30,
        custom_query_text: str = None,
        debugger: RecallDebugger = None,
    ) -> List[dict]:
        """
        三路融合检索: 文本 + 几何 + 视觉

        完整的三阶段检索流程：
        1. 三路独立召回
        2. 轻量级融合排序
        3. 完整几何精排

        Args:
            query_path: 查询STP文件路径
            query_text: 查询文本（可选，用于文本召回）
            view_dir: 视图目录（可选，用于视觉向量提取）
            top_k: 最终返回数量
            text_recall_k: 文本召回数量
            geo_recall_k: 几何召回数量
            visual_recall_k: 视觉召回数量
            fusion_top_k: 融合后保留数量
            custom_query_text: 自定义查询文本

        Returns:
            检索结果列表
        """
        start_time = time.time()

        # 解析查询零件
        query_info = parse_stp_deep(query_path)
        query_geo_vector = extract_geo_vector_standalone(query_info)

        # 构建查询文本
        if custom_query_text:
            query_text = f"""【用户查找需求】
{custom_query_text}

【参考零件特征】
{generate_retrieval_description(query_info)}"""
        elif query_text is None:
            query_text = generate_retrieval_description(query_info)

        # 准备视觉向量
        query_visual_vector = None
        view_paths = None
        if self.visual_index:
            if view_dir is None:
                view_dir = Path(query_path).parent / "views"
            view_paths = find_view_images(query_path, view_dir)

            # 如果视图不足，尝试渲染
            valid_views = [v for v in (view_paths or []) if v and Path(v).exists()]
            if len(valid_views) < 3 and HAS_RENDER:
                rendered = self._render_views(query_path, view_dir)
                if rendered:
                    view_paths = rendered

            if view_paths:
                query_visual_vector = extract_visual_vector_standalone(view_paths)

        print(f"\n[三路融合检索] 开始...")

        # ========== 阶段1: 三路召回 ==========
        print(f"  [阶段1] 三路召回: 文本{text_recall_k} + 几何{geo_recall_k} + 视觉{visual_recall_k}")
        candidates, query_info, query_geo_vector, query_visual_vector = self.three_way_recall(
            query_path=query_path,
            query_text=query_text,
            query_info=query_info,
            query_geo_vector=query_geo_vector,
            query_visual_vector=query_visual_vector,
            view_paths=view_paths,
            text_recall_k=text_recall_k,
            geo_recall_k=geo_recall_k,
            visual_recall_k=visual_recall_k,
            debugger=debugger,
        )

        if not candidates:
            print(f"  [警告] 召回结果为空")
            return []

        # ========== 阶段2: 融合排序 ==========
        print(f"  [阶段2] 融合排序: 筛选到 {fusion_top_k} 个候选")
        fusion_results = self.fusion_rerank(
            candidates=candidates,
            fusion_top_k=fusion_top_k,
            debugger=debugger,
        )

        if not fusion_results:
            print(f"  [警告] 融合排序后结果为空")
            return []

        # ========== 阶段3: 完整几何精排 ==========
        print(f"  [阶段3] 完整几何精排 ({len(fusion_results)} 个候选)")
        for i, c in enumerate(fusion_results, 1):
            print(f"    候选{i}: {Path(c['filepath']).name}")
        final_results = []

        for cand in fusion_results:
            filepath = cand['filepath']

            # 获取候选零件信息
            if filepath in self._info_cache:
                cand_info = self._info_cache[filepath]
            else:
                try:
                    cand_info = parse_stp_deep(filepath)
                    self._info_cache[filepath] = cand_info
                except Exception as e:
                    print(f"    警告: {filepath} 解析失败: {e}")
                    continue

            # 计算完整几何相似度
            geo_sim = calculate_geometric_similarity(query_info, cand_info)

            # 制造特征相似度
            q_mfg = query_info.get('mfg_features', {})
            c_mfg = cand_info.get('mfg_features', {})
            mfg_sim_val = 0.0
            if q_mfg and c_mfg:
                mfg_sim = calculate_mfg_similarity(q_mfg, c_mfg)
                mfg_sim_val = mfg_sim['overall']

            # 最终分数
            # geo_sim['overall'] 已包含制造特征权重（calculate_geometric_similarity 内部 10%）
            final_score = 0.25 * cand['fusion_score'] + 0.75 * geo_sim['overall']

            # 构建结果
            result = {
                'filepath': filepath,
                'filename': Path(filepath).name,
                'path': filepath,
                'fusion_score': cand['fusion_score'],
                'geometric_similarity': geo_sim,
                'final_score': final_score,
                'hybrid_similarity': final_score,
                'similarity': final_score,  # 兼容字段
                'recall_sources': cand.get('recall_sources', []),
                'recall_count': cand.get('recall_count', 0),
            }

            # 添加各路相似度
            if 'text_similarity' in cand:
                result['text_similarity'] = cand['text_similarity']
            if 'geo_similarity' in cand:
                result['geo_similarity'] = cand['geo_similarity']
            if 'visual_similarity' in cand:
                result['visual_similarity'] = cand['visual_similarity']

            # 添加描述
            if filepath in self._desc_cache:
                result['description'] = self._desc_cache[filepath]
            elif cand_info:
                result['description'] = generate_description(cand_info)

            final_results.append(result)

        # 按最终分数排序
        final_results.sort(key=lambda x: x['final_score'], reverse=True)

        elapsed = time.time() - start_time
        print(f"  [完成] 返回 {len(final_results[:top_k])} 个结果, 耗时 {elapsed*1000:.0f}ms")

        if debugger:
            debugger.record_final(final_results[:top_k])

        return final_results[:top_k]

    def _render_views(self, stp_path: str, view_dir: str = None) -> Optional[List[str]]:
        """
        渲染STP文件的6个视图

        Args:
            stp_path: STP文件路径
            view_dir: 视图输出目录

        Returns:
            渲染后的视图路径列表 [front, top, left, right, back, bottom]
        """
        if not HAS_RENDER:
            print("  [渲染] 渲染模块不可用")
            return None

        try:
            from step import render_6_views
            import pyvista as pv
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.StlAPI import StlAPI_Writer
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

            stp_path = Path(stp_path)
            base_name = stp_path.stem

            # 确定输出目录
            if view_dir:
                output_base = Path(view_dir)
            else:
                output_base = Path(tempfile.gettempdir()) / "stp_renders"

            output_dir = output_base / base_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # 检查是否已有渲染结果
            view_files = {
                'front': output_dir / 'front.png',
                'top': output_dir / 'top.png',
                'left': output_dir / 'left.png',
                'right': output_dir / 'right.png',
                'back': output_dir / 'back.png',
                'bottom': output_dir / 'bottom.png',
            }

            # 如果所有视图都已存在，直接返回
            if all(vf.exists() for vf in view_files.values()):
                print(f"  [渲染] 使用已有视图: {output_dir}")
                return [str(view_files[name]) for name in ['front', 'top', 'left', 'right', 'back', 'bottom']]

            # 渲染
            print(f"  [渲染] 正在渲染: {stp_path.name}...")
            temp_stl = output_dir / f"temp_{base_name}.stl"

            # STEP -> STL
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(stp_path))
            if status != 1:
                print(f"  [渲染] 无法读取STEP文件")
                return None

            reader.TransferRoots()
            shape = reader.OneShape()
            mesh = BRepMesh_IncrementalMesh(shape, 0.1)
            mesh.Perform()

            writer = StlAPI_Writer()
            writer.Write(shape, str(temp_stl))

            # PyVista渲染
            mesh = pv.read(str(temp_stl))

            views = {
                "front": (0, 0, 0),
                "back": (180, 0, 0),
                "top": (0, 90, 0),
                "bottom": (0, -90, 0),
                "left": (-90, 0, 0),
                "right": (90, 0, 0)
            }

            plotter = pv.Plotter(off_screen=True, window_size=[512, 512])
            plotter.add_mesh(mesh, color="#A7C1D1", show_edges=False,
                           edge_color="black", line_width=1,
                           smooth_shading=False, specular=0.2, ambient=0.3)
            plotter.set_background("white")

            rendered_paths = []
            for name in ['front', 'top', 'left', 'right', 'back', 'bottom']:
                az, el, roll = views[name]
                plotter.camera_position = 'xy'
                plotter.camera.azimuth = az
                plotter.camera.elevation = el
                plotter.camera.roll = roll
                plotter.reset_camera()
                plotter.screenshot(str(view_files[name]))
                rendered_paths.append(str(view_files[name]))

            plotter.close()

            # 清理临时STL
            if temp_stl.exists():
                temp_stl.unlink()

            print(f"  [渲染] 完成，生成6个视图")
            return rendered_paths

        except Exception as e:
            print(f"  [渲染] 错误: {e}")
            return None


class STPRenderer:
    """STP文件渲染器 - 将STP/STEP文件渲染为6个工程视图图片"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化渲染器

        Args:
            cache_dir: 渲染图片缓存目录，如果为None则使用系统临时目录
        """
        self.has_render = HAS_RENDER
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "stp_renders"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _step_to_stl_temp(self, step_path: str, stl_path: str) -> bool:
        """将STEP转换为临时STL文件以便PyVista渲染"""
        if not self.has_render:
            return False
        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(step_path))
            if status != 1:
                return False

            reader.TransferRoots()
            shape = reader.OneShape()

            # 划分网格 (精度设置为 0.1)
            mesh = BRepMesh_IncrementalMesh(shape, 0.1)
            mesh.Perform()

            writer = StlAPI_Writer()
            writer.Write(shape, str(stl_path))
            return True
        except Exception as e:
            print(f"  [Render Error] STEP转STL失败: {e}")
            return False

    def _render_mesh_views(self, mesh, output_dir: Path, base_name: str) -> List[str]:
        """渲染网格为6个标准视角视图"""
        view_paths = []

        # 定义6个标准视角 (方位角, 俯仰角, 翻滚角)
        views = {
            "front": (0, 0, 0),
            "back": (180, 0, 0),
            "top": (0, 90, 0),
            "bottom": (0, -90, 0),
            "left": (-90, 0, 0),
            "right": (90, 0, 0)
        }

        # 配置渲染器 (off_screen=True 表示后台运行)
        plotter = pv.Plotter(off_screen=True, window_size=[512, 512])
        plotter.add_mesh(
            mesh,
            color="#A7C1D1",
            show_edges=False,
            edge_color="black",
            line_width=1,
            smooth_shading=False,
            specular=0.2,
            ambient=0.3
        )
        plotter.set_background("white")  # 白色背景利于AI识别

        for name, (az, el, roll) in views.items():
            plotter.camera_position = 'xy'
            plotter.camera.azimuth = az
            plotter.camera.elevation = el
            plotter.camera.roll = roll
            plotter.reset_camera()

            img_path = output_dir / f"{base_name}_{name}.png"
            plotter.screenshot(str(img_path))
            view_paths.append(str(img_path))

        plotter.close()
        return view_paths

    def render_to_cache(self, stp_path: str, force: bool = False) -> Optional[List[str]]:
        """
        渲染STP文件并缓存结果

        Args:
            stp_path: STP文件路径
            force: 是否强制重新渲染

        Returns:
            6个视图图片路径列表，如果渲染失败返回None
        """
        if not self.has_render:
            print("  [Render Warning] 渲染模块未安装，跳过视觉分析")
            return None

        stp_path_obj = Path(stp_path)
        base_name = stp_path_obj.stem

        # 创建缓存子目录
        cache_subdir = self.cache_dir / base_name
        cache_subdir.mkdir(parents=True, exist_ok=True)

        # 检查是否已有缓存
        view_names = ["front", "back", "top", "bottom", "left", "right"]
        cached_views = [cache_subdir / f"{base_name}_{name}.png" for name in view_names]

        if not force and all(p.exists() for p in cached_views):
            print(f"  [Render] 使用缓存视图: {base_name}")
            return [str(p) for p in cached_views]

        print(f"  [Render] 正在渲染: {base_name}")

        # 1. 转换为STL
        temp_stl = cache_subdir / f"temp_{base_name}.stl"
        if not self._step_to_stl_temp(stp_path, temp_stl):
            return None

        # 2. 加载到PyVista并渲染
        try:
            mesh = pv.read(temp_stl)
            view_paths = self._render_mesh_views(mesh, cache_subdir, base_name)
        except Exception as e:
            print(f"  [Render Error] PyVista渲染失败: {e}")
            if temp_stl.exists():
                temp_stl.unlink()
            return None

        # 3. 清理临时STL文件
        if temp_stl.exists():
            temp_stl.unlink()

        print(f"  [Render OK] 渲染完成: {base_name}")
        return view_paths

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        清理渲染缓存

        Args:
            older_than_days: 清理多少天前的缓存，None表示清理全部
        """
        if not self.cache_dir.exists():
            return

        current_time = time.time()

        for item in self.cache_dir.iterdir():
            if item.is_dir():
                # 检查目录修改时间
                dir_time = item.stat().st_mtime
                if older_than_days is None:
                    # 清理全部
                    shutil.rmtree(item)
                    print(f"  [Cache Clean] 删除: {item.name}")
                else:
                    # 按天数清理
                    days_old = (current_time - dir_time) / (24 * 3600)
                    if days_old > older_than_days:
                        shutil.rmtree(item)
                        print(f"  [Cache Clean] 删除 {days_old:.1f}天前: {item.name}")

    def render_rotation_gif(self, stp_path: str, force: bool = False,
                            num_frames: int = 36, duration: int = 100) -> Optional[str]:
        """
        渲染STP零件的360度旋转动画GIF

        Args:
            stp_path: STP文件路径
            force: 是否强制重新渲染
            num_frames: 帧数（默认36帧，每10度一帧）
            duration: 每帧持续时间（毫秒）

        Returns:
            生成的GIF文件路径，如果渲染失败返回None
        """
        if not self.has_render:
            print("  [Render Warning] 渲染模块未安装，跳过旋转动画生成")
            return None

        stp_path_obj = Path(stp_path)
        base_name = stp_path_obj.stem

        # 创建缓存子目录
        cache_subdir = self.cache_dir / base_name
        cache_subdir.mkdir(parents=True, exist_ok=True)

        # GIF缓存路径
        gif_path = cache_subdir / f"{base_name}_rotation.gif"

        # 检查是否已有缓存
        if not force and gif_path.exists():
            print(f"  [Render] 使用缓存旋转动画: {base_name}")
            return str(gif_path)

        print(f"  [Render] 正在渲染旋转动画: {base_name} ({num_frames}帧)")

        # 1. 转换为STL
        temp_stl = cache_subdir / f"temp_{base_name}_rotation.stl"
        if not self._step_to_stl_temp(stp_path, temp_stl):
            return None

        # 2. 加载到PyVista并渲染旋转动画
        try:
            mesh = pv.read(temp_stl)
            frames = []

            # 配置渲染器
            plotter = pv.Plotter(off_screen=True, window_size=[512, 512])
            plotter.add_mesh(
                mesh,
                color="#A7C1D1",
                show_edges=False,
                edge_color="black",
                line_width=1,
                smooth_shading=False,
                specular=0.2,
                ambient=0.3
            )
            plotter.set_background("white")

            # 渲染每帧
            for i in range(num_frames):
                angle = 360 * i / num_frames
                plotter.camera_position = 'xy'
                plotter.camera.azimuth = angle
                plotter.camera.elevation = 0  # 保持水平视角
                plotter.reset_camera()

                # 渲染到内存
                img = plotter.screenshot(return_img=True)
                frames.append(img)

            plotter.close()

            # 3. 合成GIF
            imageio.mimsave(str(gif_path), frames, duration=duration, loop=0)

            print(f"  [Render OK] 旋转动画完成: {base_name} (大小: {gif_path.stat().st_size / 1024:.1f}KB)")

        except Exception as e:
            print(f"  [Render Error] 旋转动画渲染失败: {e}")
            if temp_stl.exists():
                temp_stl.unlink()
            return None

        # 4. 清理临时STL文件
        if temp_stl.exists():
            temp_stl.unlink()

        return str(gif_path)


# ============================================================
# 第七部分：视觉分析辅助函数（整合自 stp_feature）
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为base64字符串"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except (FileNotFoundError, IOError):
        return None


# ============================================================
# 新增：视图特征识别模块（多模态特征提取）
# ============================================================

class LLMVisionUnavailableError(RuntimeError):
    """LLM文本接口可用，但当前模型或服务链路无法处理图片输入。"""


class OpenAICompatibleChatGateway:
    """OpenAI 兼容聊天网关，兼容部分本地服务不支持 response_format 的情况。"""

    def __init__(self, api_key: str, base_url: str = None,
                 timeout: float = 180.0, max_retries: int = 1,
                 response_format_mode: str = "auto",
                 thinking_mode: str = "auto"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.base_url = base_url
        self.response_format_mode = response_format_mode
        self.thinking_mode = thinking_mode
        self._response_format_supported: Optional[bool] = None
        self.vision_disabled_reason: Optional[str] = None
        self.text_disabled_reason: Optional[str] = None

    @property
    def vision_available(self) -> bool:
        return self.vision_disabled_reason is None

    @staticmethod
    def _messages_contain_images(messages: list) -> bool:
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            if any(isinstance(item, dict) and item.get("type") == "image_url" for item in content):
                return True
        return False

    @staticmethod
    def _status_code(error: Exception) -> Optional[int]:
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return status if isinstance(status, int) else None

    @staticmethod
    def _response_format_unsupported(error: Exception) -> bool:
        message = str(error).lower()
        markers = (
            "response_format",
            "json_object",
            "guided decoding",
            "guided_json",
            "structured output",
        )
        return any(marker in message for marker in markers)

    def create_json_completion(self, model: str, messages: list,
                               temperature: float = 0):
        contains_images = self._messages_contain_images(messages)
        if contains_images and self.vision_disabled_reason:
            raise LLMVisionUnavailableError(self.vision_disabled_reason)
        if not contains_images and self.text_disabled_reason:
            raise RuntimeError(self.text_disabled_reason)

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.thinking_mode in ("on", "off"):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.thinking_mode == "on"
                }
            }

        use_response_format = (
            self.response_format_mode == "json"
            or (
                self.response_format_mode == "auto"
                and self._response_format_supported is not False
            )
        )
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as error:
            # 一些 vLLM/SGLang/OpenAI 兼容服务不实现 response_format。
            # auto 模式只针对该兼容性错误去掉参数重试一次，其他错误保持原样抛出。
            if (
                self.response_format_mode == "auto"
                and use_response_format
                and self._response_format_unsupported(error)
            ):
                self._response_format_supported = False
                kwargs.pop("response_format", None)
                print("  [LLM兼容] 服务不支持 response_format，已去掉该参数重试")
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as retry_error:
                    error = retry_error
                else:
                    return response

            if contains_images and self._is_visual_request_failure(error):
                self._disable_vision_or_service(model, error)
            raise error

        if use_response_format and self.response_format_mode == "auto":
            self._response_format_supported = True
        return response

    def _is_visual_request_failure(self, error: Exception) -> bool:
        status = self._status_code(error)
        message = str(error).lower()
        visual_markers = (
            "image", "vision", "multimodal", "multi-modal",
            "image_url", "pixel", "visual",
        )
        gateway_error_in_text = bool(re.search(r"\b(?:413|415|422|500|501|502|503|504)\b", message))
        return (
            status in (413, 415, 422, 500, 501, 502, 503, 504)
            or gateway_error_in_text
            or any(marker in message for marker in visual_markers)
        )

    def _disable_vision_or_service(self, model: str, visual_error: Exception):
        """视觉请求失败后只探测一次文本能力，并形成进程内熔断。"""
        status = self._status_code(visual_error)
        status_text = f"HTTP {status}" if status else str(visual_error)
        try:
            probe_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": "只回答 OK"}],
                "temperature": 0,
            }
            if self.thinking_mode in ("on", "off"):
                probe_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": self.thinking_mode == "on"
                    }
                }
            probe = self.client.chat.completions.create(**probe_kwargs)
            probe_ok = bool(
                getattr(probe, "choices", None)
                and getattr(probe.choices[0].message, "content", None)
            )
            if not probe_ok:
                raise RuntimeError("文本探测返回空响应")
        except Exception as probe_error:
            self.text_disabled_reason = (
                f"34服务器视觉请求失败（{status_text}），纯文本探测也失败: {probe_error}"
            )
            self.vision_disabled_reason = self.text_disabled_reason
            raise RuntimeError(self.text_disabled_reason) from visual_error

        self.vision_disabled_reason = (
            f"34服务器纯文本接口正常，但图片请求失败（{status_text}）；"
            "当前模型/网关不支持该视觉负载，已自动切换到纯文本精排"
        )
        print(f"  [LLM视觉熔断] {self.vision_disabled_reason}")
        raise LLMVisionUnavailableError(self.vision_disabled_reason) from visual_error


def parse_llm_json(raw_content):
    """解析本地Qwen常见的纯JSON、Markdown代码块或带思考文本的JSON响应。"""
    result = raw_content
    for _ in range(5):
        if isinstance(result, dict):
            return result
        if not isinstance(result, str):
            break

        text = result.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1)
        elif not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            break

    raise json.JSONDecodeError("未在LLM响应中找到有效JSON对象", str(raw_content), 0)


class VisionFeatureExtractor:
    """使用大模型从视图图片中提取结构化特征描述"""

    # 特征识别Prompt模板
    FEATURE_EXTRACTION_PROMPT = """你是资深机械工程师和CAD专家。请仔细分析以下STP零件的6个工程视图图片，提取零件的结构化特征。

【重要提示】
- 必须仔细观察每一张图片，不要遗漏任何细节
- 即使图片不清晰，也要基于可见信息给出最佳估计
- visual_keywords 是必需字段，必须生成5-10个关键词

【分析要求】
请从以下维度进行详细分析，并用JSON格式返回：

1. 整体形状特征：
   - 主形状类型（必须填写，如：圆柱体、方块、板类、轴类、复杂组合体、T型、L型、U型等）
   - 整体轮廓描述
   - 尺寸比例估计（长宽厚比例关系）

2. 孔洞特征：
   - 孔洞数量估计（至少给出估计值，如0、1、2、多个等）
   - 孔洞分布位置（如：中心、边缘、阵列等）
   - 孔洞类型判断（通孔、盲孔、台阶孔等）

3. 凸起/凸台特征：
   - 是否有凸台
   - 凸台数量和位置
   - 凸台形状描述

4. 凹陷/凹槽特征：
   - 是否有凹槽或凹陷
   - 凹槽类型（如：键槽、环形槽、矩形槽等）

5. 圆角/倒角特征：
   - 是否有明显圆角或倒角

6. 对称性分析：
   - 旋转对称（是否有旋转轴）
   - 镜像对称（是否有对称面）

7. 视觉关键词（必须填写5-10个）：
   - 形状类：圆柱形、方块形、板状、轴类、T型、L型、U型等
   - 特征类：多孔、单孔、法兰、凸台、凹槽、圆角等
   - 结构类：组合体、对称、非对称等

返回JSON格式（所有字段都必须填写）：
{
  "shape_type": "主形状类型（必填）",
  "shape_description": "整体轮廓详细描述",
  "dimension_ratio": {"length_width": 1.5, "width_thickness": 2.0},
  "holes": {
    "count": 0,
    "distribution": "分布描述",
    "types": []
  },
  "protrusions": {
    "has_protrusion": false,
    "count": 0,
    "description": ""
  },
  "recesses": {
    "has_recess": false,
    "types": [],
    "description": ""
  },
  "fillets_chamfers": {
    "has_fillet": false,
    "has_chamfer": false,
    "description": ""
  },
  "symmetry": {
    "rotational": false,
    "mirror": false,
    "axis_direction": ""
  },
  "special_features": [],
  "visual_keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
  "summary": "一句话总结"
}"""

    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o",
                 chat_gateway: Optional[OpenAICompatibleChatGateway] = None,
                 timeout: float = 180.0, max_retries: int = 1,
                 response_format_mode: str = "auto"):
        self.chat_gateway = chat_gateway or OpenAICompatibleChatGateway(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            response_format_mode=response_format_mode,
        )
        self.client = self.chat_gateway.client
        self.model = model
        self._feature_cache = {}  # 缓存已提取的特征

    def extract_features(self, view_paths: List[str], stp_text_features: str = None) -> Dict:
        """
        从视图图片中提取结构化特征

        Args:
            view_paths: 6个视图图片路径列表
            stp_text_features: STP解析的文本特征（可选，作为补充信息）

        Returns:
            结构化特征字典
        """
        # 过滤有效的视图路径
        valid_views = [p for p in view_paths if p and Path(p).exists()]
        if len(valid_views) < 3:
            print(f"  [特征提取] 视图不足（{len(valid_views)}/6），无法进行特征提取")
            return None

        # 检查缓存
        cache_key = str(sorted(valid_views))
        if cache_key in self._feature_cache:
            print(f"  [特征提取] 使用缓存特征")
            return self._feature_cache[cache_key]

        print(f"  [特征提取] 正在分析 {len(valid_views)} 个视图...")

        # 准备图片消息
        image_messages = prepare_view_image_messages(valid_views, include_names=True)

        # 构建完整Prompt
        prompt = self.FEATURE_EXTRACTION_PROMPT

        # 如果提供了STP文本特征，添加到Prompt中
        if stp_text_features:
            prompt += f"\n\n【STP解析的文本特征（供参考）】\n{stp_text_features}"

        # 构建消息
        content = [{"type": "text", "text": prompt}]
        content.extend(image_messages)

        messages = [{"role": "user", "content": content}]

        try:
            response = self.chat_gateway.create_json_completion(
                model=self.model,
                messages=messages,
                temperature=0,
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message.content:
                print(f"  [特征提取] LLM返回空响应")
                return None

            raw_content = response.choices[0].message.content

            # 解析 JSON：兼容代码块、思考文本以及多层JSON字符串。
            try:
                result = parse_llm_json(raw_content)
            except json.JSONDecodeError:
                result = {'visual_keywords': [], 'shape_type': '未知', 'raw': raw_content}

            # 最终检查
            if not isinstance(result, dict):
                result = {'visual_keywords': [], 'shape_type': '未知'}

            # 缓存结果
            self._feature_cache[cache_key] = result

            keywords = result.get('visual_keywords', [])
            print(f"  [特征提取] 完成，提取关键词: {keywords if keywords else []}")
            return result

        except json.JSONDecodeError as e:
            print(f"  [特征提取] JSON解析错误: {e}")
            return None
        except Exception as e:
            print(f"  [特征提取] 错误: {e}")
            return None

    def generate_feature_description(self, features: Dict) -> str:
        """
        将提取的特征转换为描述文本

        Args:
            features: 结构化特征字典

        Returns:
            特征描述文本
        """
        if not features:
            return ""

        # 确保features是字典
        if not isinstance(features, dict):
            return ""

        lines = []

        # 整体形状
        lines.append(f"视觉识别形状: {features.get('shape_type', '未知')}")
        lines.append(f"形状描述: {features.get('shape_description', '')}")

        # 尺寸比例
        dim_ratio = features.get('dimension_ratio', {})
        if isinstance(dim_ratio, dict) and dim_ratio:
            lines.append(f"视觉尺寸比例: 长/宽≈{dim_ratio.get('length_width', '?')}, 宽/厚≈{dim_ratio.get('width_thickness', '?')}")

        # 孔洞
        holes = features.get('holes', {})
        if isinstance(holes, dict):
            hole_count = holes.get('count', 0)
            if isinstance(hole_count, (int, float)) and hole_count > 0:
                lines.append(f"视觉孔洞: {hole_count}个, 分布: {holes.get('distribution', '')}, 类型: {holes.get('types', [])}")
                if holes.get('diameters_estimate'):
                    lines.append(f"孔径估计: {holes.get('diameters_estimate', [])}")

        # 凸起
        protrusions = features.get('protrusions', {})
        if isinstance(protrusions, dict) and protrusions.get('has_protrusion'):
            lines.append(f"凸台: {protrusions.get('count', 0)}个, {protrusions.get('description', '')}")

        # 凹陷
        recesses = features.get('recesses', {})
        if isinstance(recesses, dict) and recesses.get('has_recess'):
            lines.append(f"凹槽: {recesses.get('types', [])}, {recesses.get('description', '')}")

        # 圆角/倒角
        fillets = features.get('fillets_chamfers', {})
        if isinstance(fillets, dict) and (fillets.get('has_fillet') or fillets.get('has_chamfer')):
            lines.append(f"圆角/倒角: {fillets.get('description', '')}")

        # 对称性
        symmetry = features.get('symmetry', {})
        if isinstance(symmetry, dict):
            sym_types = []
            if symmetry.get('rotational'):
                sym_types.append(f"旋转对称(轴:{symmetry.get('axis_direction', '?')})")
            if symmetry.get('mirror'):
                sym_types.append("镜像对称")
            if sym_types:
                lines.append(f"对称性: {', '.join(sym_types)}")

        # 特殊特征
        special = features.get('special_features', [])
        if special and isinstance(special, list):
            lines.append(f"特殊特征: {special}")

        # 关键词
        keywords = features.get('visual_keywords', [])
        if keywords and isinstance(keywords, list):
            lines.append(f"视觉关键词: {keywords}")

        # 总结
        summary = features.get('summary', '')
        if summary and isinstance(summary, str):
            lines.append(f"视觉总结: {summary}")

        return '\n'.join(lines)

    def compare_features(self, query_features: Dict, candidate_features: Dict) -> Dict:
        """
        对比两个零件的视觉特征，生成特征匹配评分

        Args:
            query_features: 查询零件的特征
            candidate_features: 候选零件的特征

        Returns:
            特征匹配评分字典
        """
        if not query_features or not candidate_features:
            return {'overall': 0, 'details': {}}

        # 确保都是字典类型
        if not isinstance(query_features, dict) or not isinstance(candidate_features, dict):
            return {'overall': 0, 'details': {}}

        scores = {}

        # 形状类型匹配
        query_shape = query_features.get('shape_type', '') if isinstance(query_features, dict) else ''
        cand_shape = candidate_features.get('shape_type', '') if isinstance(candidate_features, dict) else ''
        shape_match = 1.0 if query_shape == cand_shape else 0.5 if query_shape and cand_shape else 0.0
        scores['shape_type'] = shape_match

        # 关键词匹配
        q_kw = query_features.get('visual_keywords', [])
        c_kw = candidate_features.get('visual_keywords', [])
        query_keywords = set(q_kw) if isinstance(q_kw, list) else set()
        cand_keywords = set(c_kw) if isinstance(c_kw, list) else set()
        if query_keywords and cand_keywords:
            keyword_overlap = len(query_keywords & cand_keywords) / max(len(query_keywords), len(cand_keywords))
            scores['keyword_overlap'] = keyword_overlap
        else:
            scores['keyword_overlap'] = 0

        # 孔洞数量匹配
        q_holes_dict = query_features.get('holes', {})
        c_holes_dict = candidate_features.get('holes', {})
        q_holes_count = q_holes_dict.get('count', 0) if isinstance(q_holes_dict, dict) else 0
        c_holes_count = c_holes_dict.get('count', 0) if isinstance(c_holes_dict, dict) else 0
        # 确保是数值
        query_holes = q_holes_count if isinstance(q_holes_count, (int, float)) else 0
        cand_holes = c_holes_count if isinstance(c_holes_count, (int, float)) else 0
        hole_sim = 1 - abs(query_holes - cand_holes) / max(query_holes, cand_holes, 1)
        scores['hole_count'] = hole_sim

        # 对称性匹配
        query_sym = query_features.get('symmetry', {})
        cand_sym = candidate_features.get('symmetry', {})
        sym_score = 0
        if isinstance(query_sym, dict) and isinstance(cand_sym, dict):
            if query_sym.get('rotational') == cand_sym.get('rotational'):
                sym_score += 0.5
            if query_sym.get('mirror') == cand_sym.get('mirror'):
                sym_score += 0.5
        scores['symmetry'] = sym_score

        # 凸起特征匹配
        q_protrusion_dict = query_features.get('protrusions', {})
        c_protrusion_dict = candidate_features.get('protrusions', {})
        query_protrusion = q_protrusion_dict.get('has_protrusion', False) if isinstance(q_protrusion_dict, dict) else False
        cand_protrusion = c_protrusion_dict.get('has_protrusion', False) if isinstance(c_protrusion_dict, dict) else False
        scores['protrusion'] = 1.0 if query_protrusion == cand_protrusion else 0.0

        # 凹槽特征匹配
        q_recess_dict = query_features.get('recesses', {})
        c_recess_dict = candidate_features.get('recesses', {})
        query_recess = q_recess_dict.get('has_recess', False) if isinstance(q_recess_dict, dict) else False
        cand_recess = c_recess_dict.get('has_recess', False) if isinstance(c_recess_dict, dict) else False
        scores['recess'] = 1.0 if query_recess == cand_recess else 0.0

        # 综合评分（加权）- 根据关键词是否有效调整权重
        has_keywords = bool(query_keywords) and bool(cand_keywords)

        if has_keywords:
            # 正常权重
            weights = {
                'shape_type': 0.25,
                'keyword_overlap': 0.20,
                'hole_count': 0.15,
                'symmetry': 0.15,
                'protrusion': 0.10,
                'recess': 0.10,
            }
        else:
            # 关键词无效时，增加其他特征的权重
            weights = {
                'shape_type': 0.35,
                'keyword_overlap': 0.0,  # 忽略关键词
                'hole_count': 0.20,
                'symmetry': 0.20,
                'protrusion': 0.125,
                'recess': 0.125,
            }

        overall = sum(scores.get(k, 0) * weights[k] for k in weights)
        scores['overall'] = round(overall, 4)

        return {'overall': scores['overall'], 'details': scores}


def prepare_image_message(image_path: str) -> Optional[dict]:
    """
    准备单张图片消息内容

    Args:
        image_path: 图片路径

    Returns:
        图片消息内容，如果图片不存在或无法读取则返回None
    """
    if not image_path or not Path(image_path).exists():
        return None

    base64_image = encode_image_to_base64(image_path)
    if base64_image is None:
        return None

    # 根据文件扩展名确定 MIME 类型
    ext = Path(image_path).suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_image}"
        }
    }


def prepare_view_image_messages(view_paths: List[str], include_names: bool = True) -> List[dict]:
    """
    准备视图图片消息内容

    Args:
        view_paths: 视图图片路径列表
        include_names: 是否包含视图名称标签

    Returns:
        消息内容列表
    """
    view_names = ['主视图', '俯视图', '左视图', '右视图', '仰视图', '后视图']
    messages = []

    for i, path in enumerate(view_paths):
        if not path or not Path(path).exists():
            continue
        base64_image = encode_image_to_base64(path)
        if base64_image is None:
            continue

        if include_names and i < len(view_names):
            messages.append({
                "type": "text",
                "text": f"\n【{view_names[i]}】"
            })

        messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

    return messages


def find_view_images(stp_path: str, view_dir: Optional[str] = None) -> List[str]:
    """
    查找STP文件对应的视图图片

    Args:
        stp_path: STP文件路径
        view_dir: 视图图片目录，如果为None则在STP文件同目录下查找

    Returns:
        6个视图图片路径列表 [主视图, 俯视图, 左视图, 右视图, 仰视图, 后视图]
    """
    stp_path_obj = Path(stp_path)
    base_name = stp_path_obj.stem

    # 确定视图目录
    if view_dir is None:
        view_dir = stp_path_obj.parent
    else:
        view_dir = Path(view_dir)

    # 视图文件名模式
    patterns = [
        # 模式1: {base_name}_front.png, {base_name}_top.png, ...
        f"{base_name}_front.png", f"{base_name}_top.png", f"{base_name}_left.png",
        f"{base_name}_right.png", f"{base_name}_bottom.png", f"{base_name}_back.png",
        # 模式2: front.png, top.png, ... (在目录下)
        "front.png", "top.png", "left.png", "right.png", "bottom.png", "back.png",
        # 模式3: {base_name}_视图名.png
        f"{base_name}_主视图.png", f"{base_name}_俯视图.png", f"{base_name}_左视图.png",
        f"{base_name}_右视图.png", f"{base_name}_仰视图.png", f"{base_name}_后视图.png",
    ]

    view_paths = [None] * 6
    view_dirs = [view_dir]

    # 如果STP文件所在目录下有以文件名命名的视图子目录，也搜索该目录
    sub_view_dir = view_dir / base_name
    if sub_view_dir.is_dir():
        view_dirs.append(sub_view_dir)

    # 搜索render_cache目录（用于存储系统中的哈希文件名零件）
    # 检查是否在storage/stp_files目录下
    stp_parent = stp_path_obj.parent
    if 'stp_files' in str(stp_parent) or 'storage' in str(stp_parent):
        # 向上查找storage目录，然后搜索render_cache
        storage_dir = stp_parent
        while storage_dir.name and storage_dir.name != 'storage':
            storage_dir = storage_dir.parent
        if storage_dir.name == 'storage':
            render_cache_dir = storage_dir / 'render_cache' / base_name
            if render_cache_dir.is_dir():
                view_dirs.insert(0, render_cache_dir)  # 优先搜索render_cache

    # 搜索每个视图
    view_positions = ['front', 'top', 'left', 'right', 'bottom', 'back']
    for idx, pos in enumerate(view_positions):
        for vdir in view_dirs:
            # 尝试不同的文件名模式
            for i in range(idx, len(patterns), 6):
                pattern = patterns[i]
                candidate = vdir / pattern
                if candidate.exists():
                    view_paths[idx] = str(candidate)
                    break
            if view_paths[idx]:
                break

    return view_paths


def has_view_images(stp_path: str, view_dir: Optional[str] = None, min_views: int = 1) -> bool:
    """
    检查STP文件是否有可用的视图图片

    Args:
        stp_path: STP文件路径
        view_dir: 视图图片目录
        min_views: 最少需要多少个视图才算有效

    Returns:
        是否有可用的视图图片
    """
    view_paths = find_view_images(stp_path, view_dir)
    available = sum(1 for p in view_paths if p is not None)
    return available >= min_views


def prepare_gif_message(gif_path: str) -> Optional[dict]:
    """
    将GIF文件编码为base64消息（用于多模态API）

    Args:
        gif_path: GIF文件路径

    Returns:
        图片消息内容，如果文件不存在或无法读取则返回None
    """
    if not gif_path or not Path(gif_path).exists():
        return None

    try:
        with open(gif_path, 'rb') as f:
            base64_gif = base64.b64encode(f.read()).decode('utf-8')

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/gif;base64,{base64_gif}"
            }
        }
    except (FileNotFoundError, IOError):
        return None


def find_rotation_gif(stp_path: str, view_dir: Optional[str] = None) -> Optional[str]:
    """
    查找STP文件对应的旋转动画GIF

    Args:
        stp_path: STP文件路径
        view_dir: 视图图片目录，如果为None则在STP文件同目录下查找

    Returns:
        旋转动画GIF路径，如果不存在返回None
    """
    stp_path_obj = Path(stp_path)
    base_name = stp_path_obj.stem

    # 确定视图目录
    if view_dir is None:
        view_dir = stp_path_obj.parent
    else:
        view_dir = Path(view_dir)

    # 可能的GIF路径
    possible_paths = [
        view_dir / base_name / f"{base_name}_rotation.gif",
        view_dir / f"{base_name}_rotation.gif",
    ]

    for gif_path in possible_paths:
        if gif_path.exists():
            return str(gif_path)

    return None


# ============================================================
# 第七部分：LLM 精排（支持多模态视觉分析）
# ============================================================

class LLMReranker:
    def __init__(self, api_key: str, base_url: str = None,
                 model: str = "gpt-4o", view_dir: Optional[str] = None,
                 timeout: float = 180.0, max_retries: int = 1,
                 response_format_mode: str = "auto",
                 thinking_mode: str = "auto"):
        """
        初始化LLM精排器

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            view_dir: 视图图片根目录（用于查找候选零件的视图）
        """
        self.chat_gateway = OpenAICompatibleChatGateway(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            response_format_mode=response_format_mode,
            thinking_mode=thinking_mode,
        )
        self.client = self.chat_gateway.client
        self.model = model
        self.view_dir = view_dir
        self._view_cache = {}  # 缓存视图路径

        endpoint_label = base_url or "OpenAI默认地址"
        print(f"  [LLM] 模型: {model}")
        print(f"  [LLM] 服务地址: {endpoint_label}")
        print(f"  [LLM] JSON响应模式: {response_format_mode}")
        print(f"  [LLM] 思考模式: {thinking_mode}")

        # 为视觉特征提取选择正确的模型
        # 阿里云需要用 qwen-vl-plus 或 qwen-vl-max 进行视觉分析
        vision_model = model
        if 'dashscope' in (base_url or '').lower() or 'aliyuncs' in (base_url or '').lower():
            # 阿里云平台：自动切换到视觉模型
            if 'vl' not in model.lower():
                # 文本模型 → 视觉模型
                vision_model = 'qwen-vl-plus'
                print(f"  [视觉特征提取] 自动切换模型: {model} → {vision_model}")

        self.feature_extractor = VisionFeatureExtractor(
            api_key,
            base_url,
            vision_model,
            chat_gateway=self.chat_gateway,
        )  # 新增：特征提取器

    @staticmethod
    def _fallback_results(candidates: list[dict], top_k: int, reason: str) -> list[dict]:
        """LLM服务器不可用或返回异常时，保留确定性的检索阶段排序。"""
        print(f"  [警告] {reason}，使用检索阶段排序")
        return [
            {
                'candidate_number': i + 1,
                'filename': c.get('filename', ''),
                'similarity_score': c.get(
                    'final_score',
                    c.get('hybrid_similarity', c.get('similarity', 0)),
                ),
                'reason': reason,
                'llm_fallback': True,
                'path': c.get('path', c.get('filepath', '')),
                'embedding_similarity': c.get('similarity', c.get('text_similarity', 0)),
                'geometric_similarity': c.get('geometric_similarity', {}),
                'hybrid_similarity': c.get(
                    'final_score',
                    c.get('hybrid_similarity', c.get('similarity', 0)),
                ),
            }
            for i, c in enumerate(candidates[:top_k])
        ]

    def rerank(self, query_desc: str, candidates: list[dict],
               top_k: int = 5,
               reference_images: List[str] = None,
               reference_dimensions: dict = None,
               custom_query_text: str = None) -> list[dict]:

        # 如果提供了自定义查找文本，用它替代或补充自动生成的描述
        if custom_query_text:
            query_desc = f"""【用户自定义查找条件】
{custom_query_text}

【STP文件自动解析信息】
{query_desc}"""

        cand_text = ""
        for i, c in enumerate(candidates, 1):
            cand_text += f"\n--- 候选零件 {i}: {c['filename']} ---\n"
            cand_text += c.get('description', c.get('filename', f'候选零件{i}')) + "\n"
            # 添加几何相似度信息
            if 'geometric_similarity' in c:
                geo = c['geometric_similarity']
                cand_text += f"几何特征相似度: {geo['overall']:.4f}\n"
                cand_text += f"  - 面类型分布: {geo['face_type_distribution']:.4f}\n"
                cand_text += f"  - 拓扑结构: {geo['topology']:.4f}\n"
                cand_text += f"  - 尺寸比例: {geo['aspect_ratio']:.4f}\n"
                cand_text += f"  - 复杂度: {geo['complexity']:.4f}\n"
                cand_text += f"  - 特征丰富度: {geo['feature_richness']:.4f}\n"
                cand_text += f"  - 紧密度: {geo['compactness']:.4f}\n"
                cand_text += f"  - 旋转对称性: {geo['rotational_symmetry']:.4f}\n"
                # 添加制造特征信息
                if 'manufacturing' in geo:
                    mfg = geo['manufacturing']
                    cand_text += f"制造特征相似度: {mfg['overall']:.4f}\n"
                    cand_text += f"  - 孔特征: {mfg.get('holes', 0):.4f}\n"
                    cand_text += f"  - 槽/型腔: {mfg.get('slots_pockets', 0):.4f}\n"
                    cand_text += f"  - 凸台: {mfg.get('bosses', 0):.4f}\n"
                    cand_text += f"  - 圆角/倒角: {mfg.get('fillets_chamfers', 0):.4f}\n"

        # 构建参考尺寸约束提示
        reference_constraint = ""
        if reference_dimensions:
            dims = reference_dimensions
            reference_constraint = f"""
【参考尺寸约束】
请优先查找符合以下目标尺寸的候选零件：
- 长度: {dims.get('length', 'N/A')}mm
- 宽度: {dims.get('width', 'N/A')}mm
- 高度: {dims.get('height', 'N/A')}mm

对于尺寸接近目标尺寸的候选零件，应在相似度评分中给予优先权重。
尺寸偏差在±10%以内的候选零件应获得更高评分。
"""

        prompt = f"""你是资深机械工程师和CAD专家。根据几何特征描述，判断哪些候选零件与查询零件最相似。
{reference_constraint}
相似性判断标准（按重要性排序）：
1. 参考尺寸匹配度（如有指定参考尺寸，优先匹配尺寸）
2. 面类型分布比例：平面/圆柱面/圆锥面/球面/环面/自由曲面的占比是否接近
3. 拓扑结构：面数、边数、欧拉特征数是否接近
4. 尺寸比例：长宽高比例关系（不是绝对尺寸）
5. 特征尺寸分布：孔径/圆角半径的种类数和分布
6. 点分布特征值比：反映质量分布对称性
7. 面类型熵：反映几何复杂度
8. 旋转对称性：反映零件的旋转对称程度
9. 紧密度：反映零件的空间利用率

重要：如果指定了参考尺寸约束，优先匹配参考尺寸。否则绝对尺寸差异不重要，形状比例才重要。同类零件不同规格应判为高相似度。

查询零件:
{query_desc}

候选零件:
{cand_text}

返回JSON，按相似度从高到低:
{{
  "rankings": [
    {{
      "candidate_number": <编号>,
      "filename": "<文件名>",
      "similarity_score": <0.0-1.0>,
      "reason": "<判断理由>"
    }}
  ]
}}"""

        try:
            response = self.chat_gateway.create_json_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
        except Exception as error:
            return self._fallback_results(
                candidates,
                top_k,
                f"LLM调用失败（model={self.model}）: {error}",
            )

        # 检查响应是否有效
        if not response.choices or not response.choices[0].message.content:
            print(f"  [警告] LLM返回空响应，使用原始排序")
            return [{'candidate_number': i+1,
                     'filename': c.get('filename', ''),
                     'similarity_score': c.get('hybrid_similarity', 0.5),
                     'reason': 'LLM响应为空',
                     'path': c.get('path', ''),
                     'embedding_similarity': c.get('similarity', 0),
                     'geometric_similarity': c.get('geometric_similarity', {}),
                     'hybrid_similarity': c.get('hybrid_similarity', 0)}
                    for i, c in enumerate(candidates[:top_k])]

        try:
            result = parse_llm_json(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"  [警告] JSON解析失败: {e}，使用原始排序")
            return [{'candidate_number': i+1,
                     'filename': c.get('filename', ''),
                     'similarity_score': c.get('hybrid_similarity', 0.5),
                     'reason': 'JSON解析失败',
                     'path': c.get('path', ''),
                     'embedding_similarity': c.get('similarity', 0),
                     'geometric_similarity': c.get('geometric_similarity', {}),
                     'hybrid_similarity': c.get('hybrid_similarity', 0)}
                    for i, c in enumerate(candidates[:top_k])]

        rankings = result.get('rankings', [])

        for r in rankings:
            idx = r['candidate_number'] - 1
            if 0 <= idx < len(candidates):
                r['path'] = candidates[idx]['path']
                r['embedding_similarity'] = candidates[idx].get('similarity', 0)
                r['geometric_similarity'] = candidates[idx].get('geometric_similarity', {})
                r['hybrid_similarity'] = candidates[idx].get('hybrid_similarity', 0)

        return rankings[:top_k]

    def rerank_with_vision(self, query_path: str, query_desc: str,
                           candidates: list[dict], top_k: int = 5,
                           query_views: List[str] = None,
                           candidate_views: List[List[str]] = None,
                           reference_images: List[str] = None,
                           reference_dimensions: dict = None,
                           custom_query_text: str = None,
                           query_rotation_gif: str = None,
                           candidate_rotation_gifs: List[str] = None) -> list[dict]:
        """
        使用多模态视觉分析进行精排

        Args:
            query_path: 查询STP文件路径
            query_desc: 查询零件的文字描述
            candidates: 候选零件列表
            top_k: 返回前K个结果
            query_views: 查询零件的视图图片路径列表（6个视图）
            candidate_views: 候选零件的视图列表，每个元素是一个候选的6个视图路径
            reference_images: 参考图片路径列表（用于指定目标零件样式/尺寸）
            reference_dimensions: 参考尺寸字典，如 {'length': 100, 'width': 60, 'height': 30}
            custom_query_text: 用户自定义查找条件文本，用于定义查找逻辑
            query_rotation_gif: 查询零件的360度旋转动画GIF路径
            candidate_rotation_gifs: 候选零件的旋转动画GIF路径列表

        Returns:
            精排后的结果列表
        """
        # 如果提供了自定义查找文本，用它替代或补充自动生成的描述
        if custom_query_text:
            query_desc = f"""【用户自定义查找条件】
{custom_query_text}

【STP文件自动解析信息】
{query_desc}"""

        # 准备查询零件的视图消息
        query_image_messages = []
        if query_views:
            query_image_messages = prepare_view_image_messages(query_views, include_names=True)

        # 准备参考图片消息
        reference_image_messages = []
        if reference_images:
            for ref_path in reference_images:
                if ref_path and os.path.exists(ref_path):
                    ref_msg = prepare_image_message(ref_path)
                    if ref_msg:
                        reference_image_messages.append(ref_msg)

        # 准备旋转动画GIF消息（3D视觉增强）
        query_gif_message = None
        candidate_gif_messages = {}

        # 处理查询零件的旋转动画
        if query_rotation_gif and Path(query_rotation_gif).exists():
            query_gif_message = prepare_gif_message(query_rotation_gif)
            if query_gif_message:
                print(f"  [3D Vision] 查询零件旋转动画已加载: {Path(query_rotation_gif).name}")

        # 处理候选零件的旋转动画
        if candidate_rotation_gifs:
            for i, gif_path in enumerate(candidate_rotation_gifs, 1):
                if gif_path and Path(gif_path).exists():
                    gif_msg = prepare_gif_message(gif_path)
                    if gif_msg:
                        candidate_gif_messages[i] = gif_msg
                        print(f"  [3D Vision] 候选零件{i}旋转动画已加载: {Path(gif_path).name}")

        has_3d_vision = query_gif_message is not None and len(candidate_gif_messages) > 0

        # 准备候选零件的视图
        candidate_view_info = {}

        # 如果提供了候选视图列表，直接使用
        if candidate_views:
            for i, (c, views) in enumerate(zip(candidates, candidate_views), 1):
                available = sum(1 for v in views if v is not None)
                candidate_view_info[i] = {
                    'paths': views,
                    'available': available
                }
        else:
            # 否则从view_dir查找
            for i, c in enumerate(candidates, 1):
                cand_path = c.get('path', '')
                if cand_path and self.view_dir:
                    # 查找候选零件的视图图片
                    if cand_path not in self._view_cache:
                        self._view_cache[cand_path] = find_view_images(cand_path, self.view_dir)
                    view_paths = self._view_cache[cand_path]
                    available_views = sum(1 for p in view_paths if p is not None)
                    candidate_view_info[i] = {
                        'paths': view_paths,
                        'available': available_views
                    }

        # 准备候选零件的文字描述
        cand_text = ""
        for i, c in enumerate(candidates, 1):
            cand_text += f"\n--- 候选零件 {i}: {c['filename']} ---\n"
            cand_text += c.get('description', c.get('filename', f'候选零件{i}')) + "\n"
            # 添加几何相似度信息
            if 'geometric_similarity' in c:
                geo = c['geometric_similarity']
                cand_text += f"几何特征相似度: {geo['overall']:.4f}\n"
                cand_text += f"  - 面类型分布: {geo['face_type_distribution']:.4f}\n"
                cand_text += f"  - 拓扑结构: {geo['topology']:.4f}\n"
                cand_text += f"  - 尺寸比例: {geo['aspect_ratio']:.4f}\n"
                cand_text += f"  - 复杂度: {geo['complexity']:.4f}\n"
                cand_text += f"  - 特征丰富度: {geo['feature_richness']:.4f}\n"
                cand_text += f"  - 紧密度: {geo['compactness']:.4f}\n"
                cand_text += f"  - 旋转对称性: {geo['rotational_symmetry']:.4f}\n"
                # 添加制造特征信息
                if 'manufacturing' in geo:
                    mfg = geo['manufacturing']
                    cand_text += f"制造特征相似度: {mfg['overall']:.4f}\n"
                    cand_text += f"  - 孔特征: {mfg.get('holes', 0):.4f}\n"
                    cand_text += f"  - 槽/型腔: {mfg.get('slots_pockets', 0):.4f}\n"
                    cand_text += f"  - 凸台: {mfg.get('bosses', 0):.4f}\n"
                    cand_text += f"  - 圆角/倒角: {mfg.get('fillets_chamfers', 0):.4f}\n"

        # 统计有视图的候选零件
        candidates_with_views = sum(1 for info in candidate_view_info.values() if info['available'] > 0)
        has_query_views = sum(1 for p in query_views if p) if query_views else 0
        has_reference_images = len(reference_image_messages) > 0

        # 构建参考尺寸约束提示
        reference_constraint = ""
        if reference_dimensions:
            dims = reference_dimensions
            reference_constraint = f"""
【参考尺寸约束】
请优先查找符合以下目标尺寸的候选零件：
- 长度: {dims.get('length', 'N/A')}mm
- 宽度: {dims.get('width', 'N/A')}mm
- 高度: {dims.get('height', 'N/A')}mm

对于尺寸接近目标尺寸的候选零件，应在相似度评分中给予优先权重。
尺寸偏差在±10%以内的候选零件应获得更高评分。
请从候选零件的描述中提取其尺寸信息（bbox），并与目标尺寸进行对比。
"""

        # 构建提示词
        if has_query_views > 0 and candidates_with_views > 0:
            # 多模态分析模式
            # 3D视觉分析提示
            three_d_prompt = ""
            if has_3d_vision:
                three_d_prompt = """
【三维视觉分析指示】
除了6个标准视图外，还提供了零件的360度旋转动画（GIF格式）。
请充分利用旋转动画来：
1. 理解零件的整体三维形态
2. 观察不同角度下的轮廓变化
3. 发现隐藏在特定视角下的特征
4. 判断曲面和复杂结构的连续性

【分析要点】
- 旋转动画展示的是同一个零件的不同角度
- 注意动画中的轮廓变化，这反映了三维形态
- 对比查询零件和候选零件的旋转动画，判断形态相似性
"""

            prompt = f"""你是资深机械工程师和CAD专家。通过对比查询零件和候选零件的文字描述及工程视图图片，判断哪些候选零件与查询零件最相似。
{reference_constraint}
【视觉分析指示】
- 查询零件视图: 有 {has_query_views} 个视图
- 候选零件视图: {candidates_with_views}/{len(candidates)} 个零件有视图
{'- 参考图片: 已提供目标零件参考图片，请优先匹配参考图片中的样式和尺寸' if has_reference_images else ''}
{'- 三维动画: 已提供360度旋转动画，请利用动画理解三维形态' if has_3d_vision else ''}
- 视觉分析重点: 整体形状、主要结构、孔洞分布、凸起/凹陷特征、圆角/倒角等
{three_d_prompt}
【相似性判断标准（按重要性排序）】
{'1. 参考尺寸匹配度: 候选零件尺寸与目标尺寸的接近程度 - 最高优先级' if reference_dimensions else ''}
{'' if not reference_dimensions else '2. '}视觉整体形状相似度: 从多视图判断整体形状是否相似{'（并与参考图片对比）' if has_reference_images else ''}
{'' if not reference_dimensions else '3. '}面类型分布比例: 平面/圆柱面/圆锥面/球面/环面/自由曲面的占比是否接近
{'' if not reference_dimensions else '4. '}拓扑结构: 面数、边数、欧拉特征数是否接近
{'' if not reference_dimensions else '5. '}尺寸比例: 长宽高比例关系（不是绝对尺寸）
{'' if not reference_dimensions else '6. '}特征尺寸分布: 孔径/圆角半径的种类数和分布
{'' if not reference_dimensions else '7. '}局部特征相似度: 孔洞、凸台、凹槽、圆角等细节特征的相似性
{'' if not reference_dimensions else '8. '}点分布特征值比: 反映质量分布对称性
{'' if not reference_dimensions else '9. '}面类型熵: 反映几何复杂度
{'' if not reference_dimensions else '10. '}旋转对称性: 反映零件的旋转对称程度

【重要提示】
{'- 如果指定了参考尺寸约束，优先匹配参考尺寸，尺寸匹配的候选应获得更高评分' if reference_dimensions else '- 优先使用视图图片进行视觉对比，视觉相似度最重要'}
{'- 参考图片中的零件样式和尺寸是查找目标，请优先匹配' if has_reference_images else ''}
{'- 对于尺寸接近目标的候选，即使形状略有差异也应给予较高评分' if reference_dimensions else '- 绝对尺寸差异不重要，形状比例才重要'}
- 同类零件不同规格应判为高相似度
- 孔洞的分布和排列方式对相似度影响很大

【查询零件】
{query_desc}

【候选零件】
{cand_text}

返回JSON，按相似度从高到低:
{{
  "rankings": [
    {{
      "candidate_number": <编号>,
      "filename": "<文件名>",
      "similarity_score": <0.0-1.0>,
      "visual_match": "视觉匹配程度描述",
      "reason": "<判断理由>"
    }}
  ]
}}"""

            # 构建多模态消息
            content = [{"type": "text", "text": prompt}]
            # 添加参考图片（如果有）
            if reference_image_messages:
                content.append({"type": "text", "text": "\n\n【目标零件参考图片 - 请优先匹配此样式和尺寸】"})
                content.extend(reference_image_messages)
            # 添加查询零件的视图图片
            if query_image_messages:
                content.append({"type": "text", "text": "\n\n【查询零件视图图片】"})
                content.extend(query_image_messages)
            # 添加候选零件的视图图片（只显示有视图的候选零件）
            for i, info in candidate_view_info.items():
                if info['available'] > 0:
                    view_messages = prepare_view_image_messages(info['paths'], include_names=True)
                    if view_messages:
                        content.append({"type": "text", "text": f"\n\n【候选零件{i}的视图图片】"})
                        content.extend(view_messages)

            # 添加旋转动画GIF（3D视觉增强）
            if query_gif_message:
                content.append({"type": "text", "text": "\n\n【查询零件360°旋转动画 - 请观察三维形态】"})
                content.append(query_gif_message)

            for i, gif_msg in candidate_gif_messages.items():
                content.append({"type": "text", "text": f"\n\n【候选零件{i}的360°旋转动画】"})
                content.append(gif_msg)

            messages = [{"role": "user", "content": content}]
        else:
            # 纯文本模式（回退）
            prompt = f"""你是资深机械工程师和CAD专家。根据几何特征描述，判断哪些候选零件与查询零件最相似。
{reference_constraint}
【注意】
- 查询零件视图: {'有' if has_query_views > 0 else '无'} {has_query_views} 个视图
- 候选零件视图: {candidates_with_views}/{len(candidates)} 个零件有视图
- 由于视图图片不足，仅基于文字描述进行判断

相似性判断标准（按重要性排序）：
{'1. 参考尺寸匹配度（如有指定参考尺寸，优先匹配尺寸）' if reference_dimensions else ''}
{'2. 面类型分布比例：平面/圆柱面/圆锥面/球面/环面/自由曲面的占比是否接近' if reference_dimensions else '1. 面类型分布比例：平面/圆柱面/圆锥面/球面/环面/自由曲面的占比是否接近'}
{'3. 拓扑结构：面数、边数、欧拉特征数是否接近' if reference_dimensions else '2. 拓扑结构：面数、边数、欧拉特征数是否接近'}
{'4. 尺寸比例：长宽高比例关系（不是绝对尺寸）' if reference_dimensions else '3. 尺寸比例：长宽高比例关系（不是绝对尺寸）'}
{'5. 特征尺寸分布：孔径/圆角半径的种类数和分布' if reference_dimensions else '4. 特征尺寸分布：孔径/圆角半径的种类数和分布'}
{'6. 点分布特征值比：反映质量分布对称性' if reference_dimensions else '5. 点分布特征值比：反映质量分布对称性'}
{'7. 面类型熵：反映几何复杂度' if reference_dimensions else '6. 面类型熵：反映几何复杂度'}
{'8. 旋转对称性：反映零件的旋转对称程度' if reference_dimensions else '7. 旋转对称性：反映零件的旋转对称程度'}
{'9. 紧密度：反映零件的空间利用率' if reference_dimensions else '8. 紧密度：反映零件的空间利用率'}

{'重要：如果指定了参考尺寸约束，优先匹配参考尺寸。否则绝对尺寸差异不重要，形状比例才重要。同类零件不同规格应判为高相似度。' if reference_dimensions else '重要：绝对尺寸差异不重要，形状比例才重要。同类零件不同规格应判为高相似度。'}

查询零件:
{query_desc}

候选零件:
{cand_text}

返回JSON，按相似度从高到低:
{{
  "rankings": [
    {{
      "candidate_number": <编号>,
      "filename": "<文件名>",
      "similarity_score": <0.0-1.0>,
      "reason": "<判断理由>"
    }}
  ]
}}"""

            messages = [{"role": "user", "content": prompt}]

        try:
            response = self.chat_gateway.create_json_completion(
                model=self.model,
                messages=messages,
                temperature=0,
            )
        except LLMVisionUnavailableError as error:
            print(f"  [视觉降级] {error}")
            return self.rerank(
                query_desc,
                candidates,
                top_k=top_k,
                reference_dimensions=reference_dimensions,
            )
        except Exception as error:
            return self._fallback_results(
                candidates,
                top_k,
                f"LLM调用失败（model={self.model}）: {error}",
            )

        # 检查响应是否有效
        if not response.choices or not response.choices[0].message.content:
            print(f"  [警告] LLM返回空响应，使用原始排序")
            # 返回原始候选排序
            return [{'candidate_number': i+1,
                     'filename': c.get('filename', ''),
                     'similarity_score': c.get('hybrid_similarity', 0.5),
                     'reason': 'LLM响应为空',
                     'path': c.get('path', ''),
                     'embedding_similarity': c.get('similarity', 0),
                     'geometric_similarity': c.get('geometric_similarity', {}),
                     'hybrid_similarity': c.get('hybrid_similarity', 0)}
                    for i, c in enumerate(candidates[:top_k])]

        try:
            result = parse_llm_json(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"  [警告] JSON解析失败: {e}，使用原始排序")
            return [{'candidate_number': i+1,
                     'filename': c.get('filename', ''),
                     'similarity_score': c.get('hybrid_similarity', 0.5),
                     'reason': 'JSON解析失败',
                     'path': c.get('path', ''),
                     'embedding_similarity': c.get('similarity', 0),
                     'geometric_similarity': c.get('geometric_similarity', {}),
                     'hybrid_similarity': c.get('hybrid_similarity', 0)}
                    for i, c in enumerate(candidates[:top_k])]

        rankings = result.get('rankings', [])

        for r in rankings:
            idx = r['candidate_number'] - 1
            if 0 <= idx < len(candidates):
                r['path'] = candidates[idx]['path']
                r['embedding_similarity'] = candidates[idx].get('similarity', 0)
                r['geometric_similarity'] = candidates[idx].get('geometric_similarity', {})
                r['hybrid_similarity'] = candidates[idx].get('hybrid_similarity', 0)
                # 添加视图信息
                r['has_views'] = candidate_view_info.get(r['candidate_number'], {}).get('available', 0)

        return rankings[:top_k]

    def rerank_with_feature_extraction(
            self, query_path: str, query_desc: str,
            candidates: list[dict], top_k: int = 5,
            query_views: List[str] = None,
            query_stp_features: str = None,
            candidate_views: List[List[str]] = None,
            renderer: 'STPRenderer' = None,
            reference_images: List[str] = None,
            reference_dimensions: dict = None,
            custom_query_text: str = None
        ) -> list[dict]:
        """
        特征增强的多模态精排：先提取查询件特征，再进行多模态对比

        Args:
            query_path: 查询STP文件路径
            query_desc: 查询零件的文字描述
            candidates: 候选零件列表
            top_k: 返回前K个结果
            query_views: 查询零件的视图图片路径列表
            query_stp_features: STP解析的文本特征
            candidate_views: 候选零件的视图列表
            renderer: 渲染器实例（用于自动渲染）
            reference_images: 参考图片路径列表（用于指定目标零件样式/尺寸）
            reference_dimensions: 参考尺寸字典，如 {'length': 100, 'width': 60, 'height': 30}
            custom_query_text: 用户自定义查找条件文本，用于定义查找逻辑

        Returns:
            精排后的结果列表
        """
        print(f"\n  [特征增强精排] 开始分析...")

        # 如果提供了自定义查找文本，用它替代或补充自动生成的描述
        if custom_query_text:
            query_desc = f"""【用户自定义查找条件】
{custom_query_text}

【STP文件自动解析信息】
{query_desc}"""
            print(f"  [自定义查找条件] {custom_query_text[:100]}...")

        # 准备参考图片消息
        reference_image_messages = []
        if reference_images:
            for ref_path in reference_images:
                if ref_path and os.path.exists(ref_path):
                    ref_msg = prepare_image_message(ref_path)
                    if ref_msg:
                        reference_image_messages.append(ref_msg)
            if reference_image_messages:
                print(f"  [参考图片] 已加载 {len(reference_image_messages)} 个参考图片")

        # ========== 阶段1: 提取查询零件的视觉特征 ==========
        query_features = None
        query_feature_desc = ""

        if query_views and sum(1 for v in query_views if v) >= 3:
            print(f"\n  [阶段1] 提取查询零件视觉特征...")
            query_features = self.feature_extractor.extract_features(
                query_views, query_stp_features
            )
            if query_features and isinstance(query_features, dict):
                query_feature_desc = self.feature_extractor.generate_feature_description(query_features)
                print(f"  [阶段1] 查询零件特征关键词: {query_features.get('visual_keywords', [])}")

        # ========== 阶段2: 提取候选零件的视觉特征 ==========
        candidate_features = {}
        candidate_feature_descs = {}
        candidate_view_info = {}

        print(f"\n  [阶段2] 提取候选零件视觉特征...")
        extract_candidate_features = self.chat_gateway.vision_available
        if not extract_candidate_features:
            print("  [阶段2] 已触发视觉熔断，跳过所有候选的视觉特征请求")

        for i, c in enumerate(candidates, 1):
            cand_path = c.get('path', '')
            cand_views = None

            # 获取候选零件视图
            if candidate_views and i <= len(candidate_views):
                cand_views = candidate_views[i-1]
            elif cand_path:
                if cand_path not in self._view_cache:
                    self._view_cache[cand_path] = find_view_images(cand_path, self.view_dir)
                cand_views = self._view_cache[cand_path]

                # 如果视图不足且有渲染器，尝试渲染
                available = sum(1 for v in cand_views if v) if cand_views else 0
                if available < 3 and renderer:
                    cand_name = Path(c['filename']).stem
                    rendered = renderer.render_to_cache(cand_path)
                    if rendered:
                        cand_views = rendered
                        available = 6

            # 记录视图信息
            available_views = sum(1 for v in cand_views if v) if cand_views else 0
            candidate_view_info[i] = {'paths': cand_views, 'available': available_views}

            # 提取候选零件特征
            if available_views >= 3 and extract_candidate_features:
                cand_features = self.feature_extractor.extract_features(cand_views)
                if cand_features and isinstance(cand_features, dict):
                    candidate_features[i] = cand_features
                    candidate_feature_descs[i] = self.feature_extractor.generate_feature_description(cand_features)
                    keywords = cand_features.get('visual_keywords', [])
                    print(f"    候选{i}: {c['filename'][:20]} - 关键词: {keywords[:3] if keywords else []}")
                if not self.chat_gateway.vision_available:
                    extract_candidate_features = False
                    print("  [阶段2] 视觉请求失败，停止剩余候选特征提取")

        # ========== 阶段3: 特征对比评分 ==========
        feature_match_scores = {}

        # 检查查询特征是否有效
        query_keywords = query_features.get('visual_keywords', []) if query_features else []
        query_shape = query_features.get('shape_type', '') if query_features else ''
        query_features_valid = bool(query_keywords) or (query_shape and query_shape not in ['未知', '未知类型', ''])

        if query_features and isinstance(query_features, dict) and query_features_valid:
            print(f"\n  [阶段3] 特征对比评分...")
            print(f"    查询零件关键词: {query_keywords[:5] if query_keywords else '无'}")
            for i, cand_feat in candidate_features.items():
                if isinstance(cand_feat, dict):
                    match_result = self.feature_extractor.compare_features(query_features, cand_feat)
                    feature_match_scores[i] = match_result
                    print(f"    候选{i}: 特征匹配度={match_result['overall']:.4f}")
        elif query_features:
            print(f"\n  [阶段3] 特征对比评分... (跳过 - 查询零件特征提取不完整)")
            print(f"    提示: 查询零件的关键词为空，建议检查视图图片质量")
        else:
            print(f"\n  [阶段3] 特征对比评分... (跳过 - 查询零件无视觉特征)")

        # ========== 阶段4: 多模态LLM精排 ==========
        if not self.chat_gateway.vision_available:
            print(f"\n  [阶段4] 视觉接口不可用，自动切换为纯文本LLM精排...")
            return self.rerank(
                query_desc,
                candidates,
                top_k=top_k,
                reference_dimensions=reference_dimensions,
                custom_query_text=custom_query_text,
            )

        print(f"\n  [阶段4] 多模态LLM精排...")

        # 准备候选零件文字描述（包含特征信息）
        cand_text = ""
        for i, c in enumerate(candidates, 1):
            cand_text += f"\n--- 候选零件 {i}: {c['filename']} ---\n"
            # 安全获取 description，如果不存在则使用 filename
            cand_text += c.get('description', c.get('filename', f'候选零件{i}')) + "\n"

            # 添加几何相似度信息
            if 'geometric_similarity' in c:
                geo = c['geometric_similarity']
                cand_text += f"几何特征相似度: {geo['overall']:.4f}\n"
                # 添加制造特征信息
                if 'manufacturing' in geo:
                    mfg = geo['manufacturing']
                    cand_text += f"制造特征相似度: {mfg['overall']:.4f}\n"

            # 添加视觉特征描述
            if i in candidate_feature_descs:
                cand_text += f"\n【视觉识别特征】\n{candidate_feature_descs[i]}\n"

            # 添加特征匹配评分
            if i in feature_match_scores:
                match = feature_match_scores[i]
                cand_text += f"特征匹配评分: {match['overall']:.4f}\n"
                details = match.get('details', {})
                cand_text += f"  - 形状匹配: {details.get('shape_type', 0):.2f}\n"
                cand_text += f"  - 关键词重叠: {details.get('keyword_overlap', 0):.2f}\n"
                cand_text += f"  - 孔洞匹配: {details.get('hole_count', 0):.2f}\n"
                cand_text += f"  - 对称性匹配: {details.get('symmetry', 0):.2f}\n"

        # 构建增强的Prompt
        # 构建参考尺寸约束提示
        reference_constraint = ""
        if reference_dimensions:
            dims = reference_dimensions
            reference_constraint = f"""
【参考尺寸约束】
请优先查找符合以下目标尺寸的候选零件：
- 长度: {dims.get('length', 'N/A')}mm
- 宽度: {dims.get('width', 'N/A')}mm
- 高度: {dims.get('height', 'N/A')}mm

对于尺寸接近目标尺寸的候选零件，应在相似度评分中给予优先权重。
尺寸偏差在±10%以内的候选零件应获得更高评分。
请从候选零件的描述中提取其尺寸信息（bbox），并与目标尺寸进行对比。
"""

        has_reference_images = len(reference_image_messages) > 0

        prompt = f"""你是资深机械工程师和CAD专家。请综合分析以下信息，判断候选零件与查询零件的相似度。
{reference_constraint}
【分析数据来源】
1. STP解析的几何特征（面类型、拓扑、尺寸比例等）
2. 大模型视觉识别特征（从6视图图片提取的结构化特征）
3. 特征对比评分（形状匹配、关键词重叠、孔洞匹配、对称性匹配）
4. 视觉直观对比（如有视图图片）
{'5. 参考图片（如有参考图片，请优先匹配参考图片中的样式和尺寸）' if has_reference_images else ''}

【查询零件】
{query_desc}

【查询零件视觉识别特征】
{query_feature_desc if query_feature_desc else "（视图不足，未提取）"}

【候选零件】
{cand_text}

【相似度判断标准】（权重从高到低）
{'1. 参考尺寸匹配度（尺寸接近目标的候选优先）- 最高权重' if reference_dimensions else ''}
{'2. 视觉特征匹配评分（形状类型、关键词重叠、孔洞数量等）- 权重30%' if reference_dimensions else '1. 视觉特征匹配评分（形状类型、关键词重叠、孔洞数量等）- 权重30%'}
{'3. 整体视觉形状相似度（从视图图片直观判断，并与参考图片对比）- 权重25%' if has_reference_images else ('3. 整体视觉形状相似度（从视图图片直观判断）- 权重25%' if reference_dimensions else '2. 整体视觉形状相似度（从视图图片直观判断）- 权重25%')}
{'4. 几何特征相似度（面类型分布、拓扑结构）- 权重20%' if reference_dimensions else '3. 几何特征相似度（面类型分布、拓扑结构）- 权重20%'}
{'5. 尺寸比例相似度（长宽厚比例）- 权重15%' if reference_dimensions else '4. 尺寸比例相似度（长宽厚比例）- 权重15%'}
{'6. 特殊特征匹配（凸台、凹槽、对称性）- 权重10%' if reference_dimensions else '5. 特殊特征匹配（凸台、凹槽、对称性）- 权重10%'}

【重要提示】
{'- 如果指定了参考尺寸约束，优先匹配参考尺寸，尺寸匹配的候选应获得更高评分' if reference_dimensions else '- 优先参考视觉识别特征和特征匹配评分'}
{'- 参考图片中的零件样式和尺寸是查找目标，请优先匹配' if has_reference_images else ''}
{'- 对于尺寸接近目标的候选，即使形状略有差异也应给予较高评分' if reference_dimensions else '- 绝对尺寸不重要，形状比例才重要'}
- 同类零件不同规格应判为高相似度

请返回JSON格式的排名结果：
{{
  "rankings": [
    {{
      "candidate_number": <编号>,
      "filename": "<文件名>",
      "similarity_score": <0.0-1.0>,
      "feature_match_score": "<特征匹配评分（如有）>",
      "visual_match": "<视觉匹配程度描述>",
      "key_matches": ["<主要匹配点列表>"],
      "reason": "<综合判断理由>"
    }}
  ]
}}"""

        # 构建多模态消息
        content = [{"type": "text", "text": prompt}]

        # 添加参考图片（如果有）
        if reference_image_messages:
            content.append({"type": "text", "text": "\n\n【目标零件参考图片 - 请优先匹配此样式和尺寸】"})
            content.extend(reference_image_messages)

        # 添加查询零件视图图片
        query_image_messages = []
        if query_views:
            query_image_messages = prepare_view_image_messages(query_views, include_names=True)
        if query_image_messages:
            content.append({"type": "text", "text": "\n\n【查询零件视图图片】"})
            content.extend(query_image_messages)

        # 添加候选零件视图图片（只添加有视图且有特征的候选）
        for i, info in candidate_view_info.items():
            if info['available'] >= 3 and i in candidate_features:
                view_messages = prepare_view_image_messages(info['paths'], include_names=True)
                if view_messages:
                    content.append({"type": "text", "text": f"\n\n【候选零件{i}视图图片】"})
                    content.extend(view_messages)

        messages = [{"role": "user", "content": content}]

        try:
            response = self.chat_gateway.create_json_completion(
                model=self.model,
                messages=messages,
                temperature=0,
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message.content:
                print(f"  [阶段4] LLM返回空响应，回退到普通精排")
                return self.rerank_with_vision(
                    query_path, query_desc, candidates, top_k,
                    query_views, candidate_views,
                    reference_images, reference_dimensions,
                    custom_query_text
                )

            result = parse_llm_json(response.choices[0].message.content)
            rankings = result.get('rankings', [])

            # 补充信息
            for r in rankings:
                idx = r['candidate_number'] - 1
                if 0 <= idx < len(candidates):
                    r['path'] = candidates[idx]['path']
                    r['embedding_similarity'] = candidates[idx].get('similarity', 0)
                    r['geometric_similarity'] = candidates[idx].get('geometric_similarity', {})
                    r['hybrid_similarity'] = candidates[idx].get('hybrid_similarity', 0)
                    r['has_views'] = candidate_view_info.get(r['candidate_number'], {}).get('available', 0)
                    # 添加特征匹配评分
                    if r['candidate_number'] in feature_match_scores:
                        score_data = feature_match_scores[r['candidate_number']]
                        # 确保 score_data 是字典且包含有效数值
                        if isinstance(score_data, dict):
                            overall = score_data.get('overall', 0)
                            # 确保整体评分是数值
                            if isinstance(overall, (int, float)):
                                r['feature_match_score'] = overall
                            else:
                                r['feature_match_score'] = 0
                            r['feature_match_details'] = score_data.get('details', {})
                        else:
                            r['feature_match_score'] = 0
                    # 添加视觉特征
                    if r['candidate_number'] in candidate_features:
                        r['visual_features'] = candidate_features[r['candidate_number']]

            print(f"  [阶段4] LLM精排完成")
            return rankings[:top_k]

        except json.JSONDecodeError as e:
            print(f"  [阶段4] JSON解析错误: {e}")
            # 回退到普通多模态精排
            return self.rerank_with_vision(
                query_path, query_desc, candidates, top_k,
                query_views, candidate_views,
                reference_images, reference_dimensions,
                custom_query_text
            )
        except LLMVisionUnavailableError as e:
            print(f"  [阶段4] 视觉降级: {e}")
            return self.rerank(
                query_desc,
                candidates,
                top_k=top_k,
                reference_dimensions=reference_dimensions,
            )
        except Exception as e:
            print(f"  [阶段4] 错误: {e}")
            return self.rerank_with_vision(
                query_path, query_desc, candidates, top_k,
                query_views, candidate_views,
                reference_images, reference_dimensions,
                custom_query_text
            )


# ============================================================
# 第八部分：Word报告生成器
# ============================================================

class WordReportGenerator:
    """生成STP检索结果的Word文档报告"""

    def __init__(self):
        self.doc = Document()

    def _set_style(self):
        """设置文档样式"""
        # 默认字体
        style = self.doc.styles['Normal']
        style.font.name = '宋体'
        style.font.size = Pt(10)

    def _add_title(self, text: str, level: int = 1):
        """添加标题"""
        heading = self.doc.add_heading(text, level=level)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def _add_paragraph(self, text: str, bold: bool = False, font_size: int = 10):
        """添加段落"""
        p = self.doc.add_paragraph(text)
        if p.runs:  # 只有当有文本内容时才设置样式
            if bold:
                p.runs[0].bold = True
            p.runs[0].font.size = Pt(font_size)

    def _add_table_row(self, table, data: list, is_header: bool = False):
        """添加表格行"""
        row_cells = table.add_row().cells
        for i, cell_text in enumerate(data):
            cell = row_cells[i]
            cell.text = str(cell_text)
            if is_header:
                cell.paragraphs[0].runs[0].bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(10)
            else:
                cell.paragraphs[0].runs[0].font.size = Pt(9)

    def _add_gif_frame(self, gif_path: str, caption: str = ""):
        """
        将GIF的第一帧添加到文档中

        Args:
            gif_path: GIF文件路径
            caption: 图片标题
        """
        try:
            from PIL import Image
            import io
            import tempfile

            # 打开GIF并获取第一帧
            with Image.open(gif_path) as img:
                # 跳转到第一帧
                img.seek(0)

                # 转换为RGB模式（如果需要）
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # 保存为临时PNG文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    img.save(tmp.name, 'PNG')
                    tmp_path = tmp.name

                # 添加到文档
                paragraph = self.doc.add_paragraph()
                run = paragraph.add_run()
                run.add_picture(tmp_path, width=Inches(3.0))

                if caption:
                    cap_para = self.doc.add_paragraph(caption)
                    cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

                # 删除临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except ImportError:
            self._add_paragraph(f"  [需要PIL库来处理GIF图像]")
        except Exception as e:
            self._add_paragraph(f"  [GIF处理错误: {e}]")

    def generate_report(self, query_path: str, query_info: dict, results: list[dict],
                        use_llm_rerank: bool = True, save_path: str = None,
                        query_rotation_gif: str = None, candidate_rotation_gifs: dict = None) -> str:
        """
        生成检索结果报告

        Args:
            query_path: 查询文件路径
            query_info: 查询零件信息
            results: 检索结果列表
            use_llm_rerank: 是否使用了LLM精排
            save_path: 保存路径，如果为None则自动生成
            query_rotation_gif: 查询零件的3D旋转GIF路径
            candidate_rotation_gifs: 候选零件的3D旋转GIF路径字典 {filename: gif_path}

        Returns:
            生成的文件路径
        """
        self._set_style()

        # 文档标题
        self._add_title("STP零件相似度检索报告", 0)
        self._add_paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.doc.add_paragraph()  # 空行

        # 查询零件信息
        self._add_title("1. 查询零件信息", 1)
        self._add_paragraph(f"文件名: {query_info['filename']}")
        self._add_paragraph(f"文件路径: {query_path}")
        self.doc.add_paragraph()

        # 查询零件几何特征
        self._add_title("1.1 几何特征", 2)
        self._add_paragraph(f"面数: {query_info['num_faces']}")
        self._add_paragraph(f"边数: {query_info['num_edges']}")
        self._add_paragraph(f"顶点数: {query_info['num_vertices']}")
        self._add_paragraph(f"欧拉数: {query_info['euler']}")
        self.doc.add_paragraph()

        self._add_title("1.2 面类型分布", 2)
        ft = query_info['face_types']
        total_f = max(query_info['num_faces'], 1)
        for stype, count in ft.items():
            pct = count / total_f * 100
            self._add_paragraph(f"  {stype}: {count}个 ({pct:.1f}%)")
        self.doc.add_paragraph()

        self._add_title("1.3 尺寸信息", 2)
        dims = query_info['bbox_dims']
        self._add_paragraph(f"边界框尺寸: {dims[0]} x {dims[1]} x {dims[2]} mm")
        ar = query_info['aspect_ratios']
        self._add_paragraph(f"尺寸比例: 长宽比={ar[0]:.2f}, 高宽比={ar[1]:.2f}")
        self.doc.add_paragraph()

        # 检索结果摘要
        self._add_title("2. 检索结果", 1)
        self._add_paragraph(f"检索模式: {'混合检索 (Embedding + 几何特征 + LLM精排)' if use_llm_rerank else '混合检索 (Embedding + 几何特征)'}")
        self._add_paragraph(f"返回结果数: {len(results)}")
        self.doc.add_paragraph()

        # 结果详情表格
        if results:
            self._add_title("2.1 相似度排名", 2)

            # 创建结果表格（添加视觉相似度列）
            if use_llm_rerank:
                table = self.doc.add_table(rows=1, cols=7)
                table.style = 'Light Grid Accent 1'
                self._add_table_row(table, ['排名', '文件名', 'LLM评分', 'Embedding', '几何', '视觉', '混合'], is_header=True)
            else:
                table = self.doc.add_table(rows=1, cols=4)
                table.style = 'Light Grid Accent 1'
                self._add_table_row(table, ['排名', '文件名', '相似度', '相似度类型'], is_header=True)

            for i, r in enumerate(results, 1):
                if use_llm_rerank:
                    geo_sim = r.get('geometric_similarity', {})
                    geo_overall = geo_sim.get('overall', 0) if geo_sim else 0
                    visual_sim = r.get('visual_similarity', 0)
                    if not isinstance(visual_sim, (int, float)):
                        visual_sim = 0
                    self._add_table_row(table, [
                        i,
                        r['filename'],
                        f"{r.get('similarity_score', 0):.3f}",
                        f"{r.get('embedding_similarity', 0):.4f}",
                        f"{geo_overall:.4f}",
                        f"{visual_sim:.4f}",
                        f"{r.get('hybrid_similarity', 0):.4f}"
                    ])
                else:
                    sim = r.get('hybrid_similarity', r.get('similarity', 0))
                    sim_type = "混合" if 'hybrid_similarity' in r else "嵌入"
                    self._add_table_row(table, [i, r['filename'], f"{sim:.4f}", sim_type])
            self.doc.add_paragraph()

            # 详细结果分析
            self._add_title("2.2 详细分析", 2)

            for i, r in enumerate(results[:5], 1):  # 只显示前5个的详情
                self._add_paragraph(f"{'='*60}", bold=True)
                self._add_paragraph(f"排名 #{i}: {r['filename']}", bold=True, font_size=11)

                # 相似度信息
                if use_llm_rerank:
                    self._add_paragraph(f"LLM评分: {r.get('similarity_score', 0):.3f}")
                    self._add_paragraph(f"判断理由: {r.get('reason', 'N/A')}")

                emb_sim = r.get('embedding_similarity', r.get('similarity', 0))
                geo_sim = r.get('geometric_similarity', {})
                hybrid_sim = r.get('hybrid_similarity', 0)
                visual_sim = r.get('visual_similarity', 0)
                feature_match = r.get('feature_match_score', 0)

                self._add_paragraph(f"Embedding相似度: {emb_sim:.4f}")
                if geo_sim:
                    self._add_paragraph(f"几何特征相似度:")
                    self._add_paragraph(f"  - 综合评分: {geo_sim.get('overall', 0):.4f}")
                    self._add_paragraph(f"  - 面类型分布: {geo_sim.get('face_type_distribution', 0):.4f}")
                    self._add_paragraph(f"  - 拓扑结构: {geo_sim.get('topology', 0):.4f}")
                    self._add_paragraph(f"  - 尺寸比例: {geo_sim.get('aspect_ratio', 0):.4f}")
                    self._add_paragraph(f"  - 点分布: {geo_sim.get('point_distribution', 0):.4f}")
                    self._add_paragraph(f"  - 复杂度: {geo_sim.get('complexity', 0):.4f}")
                    self._add_paragraph(f"  - 特征丰富度: {geo_sim.get('feature_richness', 0):.4f}")
                    self._add_paragraph(f"  - 紧密度: {geo_sim.get('compactness', 0):.4f}")
                    self._add_paragraph(f"  - 旋转对称性: {geo_sim.get('rotational_symmetry', 0):.4f}")

                # 添加视觉相似度信息
                if isinstance(visual_sim, (int, float)) and visual_sim > 0:
                    self._add_paragraph(f"视觉相似度: {visual_sim:.4f}")

                # 添加特征匹配评分
                if isinstance(feature_match, (int, float)) and feature_match > 0:
                    self._add_paragraph(f"特征匹配评分: {feature_match:.4f}")
                    key_matches = r.get('key_matches', [])
                    if key_matches:
                        self._add_paragraph(f"关键匹配项: {', '.join(key_matches)}")

                # 添加视觉关键词
                visual_features = r.get('visual_features', {})
                if visual_features and isinstance(visual_features, dict):
                    keywords = visual_features.get('visual_keywords', [])
                    if keywords:
                        self._add_paragraph(f"视觉关键词: {', '.join(keywords[:5])}")

                if hybrid_sim:
                    self._add_paragraph(f"混合相似度: {hybrid_sim:.4f}")

                self.doc.add_paragraph()

        # 添加3D旋转动画GIF部分
        if query_rotation_gif or candidate_rotation_gifs:
            self._add_title("3. 3D旋转动画展示", 1)

            # 添加查询零件的GIF
            if query_rotation_gif and Path(query_rotation_gif).exists():
                self._add_paragraph("查询零件3D旋转动画:", bold=True)
                try:
                    # 将GIF第一帧添加到报告中
                    self._add_gif_frame(query_rotation_gif, "查询零件")
                except Exception as e:
                    self._add_paragraph(f"  [无法添加GIF图像: {e}]")

            # 添加候选零件的GIF（前3个）
            if candidate_rotation_gifs:
                self._add_paragraph("")
                self._add_paragraph("候选零件3D旋转动画:", bold=True)
                shown_count = 0
                for filename, gif_path in candidate_rotation_gifs.items():
                    if shown_count >= 3:
                        break
                    if gif_path and Path(gif_path).exists():
                        self._add_paragraph(f"  {filename}:")
                        try:
                            self._add_gif_frame(gif_path, filename)
                            shown_count += 1
                        except Exception as e:
                            self._add_paragraph(f"    [无法添加GIF: {e}]")

            self.doc.add_paragraph()

        # 保存文件
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            query_name = Path(query_path).stem
            save_path = os.path.join(os.path.dirname(query_path),
                                    f"STP检索报告_{query_name}_{timestamp}.docx")

        self.doc.save(save_path)
        return save_path


# ============================================================
# 第九部分：完整引擎
# ============================================================

class STPSearchEngine:
    def __init__(self, api_key: str, base_url: str = None,
                 embedding_model: str = "text-embedding-3-large",
                 llm_model: str = "gpt-4o",
                 db_path: str = "./chroma_parts",
                 use_faiss: bool = False,
                 view_dir: Optional[str] = None,
                 render_cache_dir: Optional[str] = None,
                 enable_render: bool = True,
                 llm_api_key: str = None,
                 llm_base_url: str = None,
                 llm_timeout: float = 180.0,
                 llm_max_retries: int = 1,
                 llm_response_format: str = "auto",
                 llm_thinking: str = "auto"):
        """
        初始化STP检索引擎

        Args:
            api_key: API密钥
            base_url: API基础URL
            embedding_model: 嵌入模型名称
            llm_model: LLM模型名称
            db_path: 向量数据库路径
            use_faiss: 是否使用FAISS索引（后端索引格式）
            view_dir: 视图图片根目录（用于视觉精排）
            render_cache_dir: 渲染缓存目录
            enable_render: 是否启用自动渲染功能
            llm_api_key: LLM专用API密钥；不提供时沿用api_key
            llm_base_url: LLM专用服务地址；不提供时沿用base_url
            llm_timeout: LLM单次请求超时秒数
            llm_max_retries: OpenAI客户端对LLM请求的自动重试次数
            llm_response_format: JSON响应兼容模式（auto/json/none）
            llm_thinking: Qwen思考模式（auto/on/off）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.view_dir = view_dir
        self.enable_render = enable_render
        self.index = EmbeddingIndex(api_key, base_url, db_path, embedding_model, use_faiss=use_faiss)
        resolved_llm_key = llm_api_key if llm_api_key is not None else api_key
        resolved_llm_url = llm_base_url if llm_base_url is not None else base_url
        self.llm_api_key = resolved_llm_key
        self.llm_base_url = resolved_llm_url
        self.reranker = LLMReranker(
            resolved_llm_key,
            resolved_llm_url,
            llm_model,
            view_dir=view_dir,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
            response_format_mode=llm_response_format,
            thinking_mode=llm_thinking,
        )
        # 初始化渲染器
        self.renderer = STPRenderer(cache_dir=render_cache_dir) if enable_render else None

    def build_index(self, directory: str, view_dir: str = None):
        """
        构建索引

        Args:
            directory: STP文件目录
            view_dir: 视图图片目录（用于视觉向量提取）
        """
        self.index.build_index(directory, view_dir=view_dir)

    def search(self, query_path: str,
               coarse_top: int = 20,
               final_top: int = 5,
               use_llm_rerank: bool = True,
               use_vision: bool = True,
               use_geometric: bool = True,
               use_feature_extraction: bool = True,
               use_three_way: bool = False,
               use_3d_vision: bool = False,
               auto_render: bool = True,
               save_report: bool = True,
               report_path: str = None,
               query_views: List[str] = None,
               reference_images: List[str] = None,
               reference_dimensions: dict = None,
               custom_query_text: str = None,
               debug_recall: str = None,
               text_recall_k: int = 150,  # 三路融合参数
               geo_recall_k: int = 150,
               visual_recall_k: int = 150,
               fusion_top_k: int = 30) -> list[dict]:
        """
        检索相似零件

        Args:
            query_path: 查询STP文件路径
            coarse_top: 粗检召回数量
            final_top: 最终返回数量
            use_llm_rerank: 是否使用LLM精排
            use_vision: 是否使用视觉分析（多模态），仅在use_llm_rerank=True时有效
            use_geometric: 是否使用几何特征相似度
            use_feature_extraction: 是否使用特征提取增强（先识别查询件特征再对比）
            use_three_way: 是否使用三路融合检索（独立召回+融合排序+几何精排）
            use_3d_vision: 是否使用三维旋转动画（GIF）进行视觉增强
            auto_render: 如果视图不存在，是否自动渲染（需要pyvista和pythonocc）
            save_report: 是否生成Word报告
            report_path: Word报告保存路径
            query_views: 查询零件的视图路径列表（6个视图），如果为None则自动查找或渲染
            reference_images: 参考图片路径列表（如1.png, 2.jpg），用于指定目标零件样式/尺寸
            reference_dimensions: 参考尺寸字典，如 {'length': 100, 'width': 60, 'height': 30}
            custom_query_text: 用户自定义查找条件文本，用于定义查找逻辑（如"查找U形零件，尺寸100x60x30mm，有2个孔")
            text_recall_k: 文本召回数量（三路融合模式）
            geo_recall_k: 几何召回数量（三路融合模式）
            visual_recall_k: 视觉召回数量（三路融合模式）
            fusion_top_k: 融合排序后保留数量（三路融合模式）

        Returns:
            精排后的结果列表
        """
        query_name = Path(query_path).name
        debugger = RecallDebugger(debug_recall) if debug_recall else None
        print(f"\n{'='*60}")
        print(f"查询: {query_name}")
        if custom_query_text:
            print(f"自定义查找条件: {custom_query_text}")
        if reference_images:
            print(f"参考图片: {reference_images}")
        if reference_dimensions:
            print(f"参考尺寸: {reference_dimensions.get('length')}x{reference_dimensions.get('width')}x{reference_dimensions.get('height')}mm")
        if use_three_way:
            print(f"检索模式: 三路融合检索（文本+几何+视觉独立召回+融合排序）")
        print(f"{'='*60}")

        # 获取查询零件信息
        query_info = parse_stp_deep(query_path)

        # 阶段1: 混合检索 或 三路融合检索
        if use_three_way:
            print(f"\n[阶段1] 三路融合检索...")
            t0 = time.time()
            coarse = self.index.search_three_way(
                query_path=query_path,
                top_k=coarse_top,
                text_recall_k=text_recall_k,
                geo_recall_k=geo_recall_k,
                visual_recall_k=visual_recall_k,
                fusion_top_k=fusion_top_k,
                custom_query_text=custom_query_text,
                view_dir=self.view_dir,
                debugger=debugger,
            )
        else:
            print(f"\n[阶段1] 混合检索 (Embedding + 几何特征) top-{coarse_top}...")
            if custom_query_text:
                print(f"  [自定义查询] 使用自定义文本进行Embedding检索: {custom_query_text[:50]}...")
            t0 = time.time()
            coarse = self.index.search(query_path, top_k=coarse_top, include_geometric=use_geometric,
                                        custom_query_text=custom_query_text)

        # 排除自身和测试零件（文件名匹配）
        excluded_by_name = lambda fn: fn == query_name or '测试件' in Path(fn).stem
        excluded = [r for r in coarse if excluded_by_name(r['filename'])]
        coarse = [r for r in coarse if not excluded_by_name(r['filename'])]

        # 诊断：显示排除信息
        if excluded:
            print(f"  [排除] 排除了 {len(excluded)} 个零件: {[r['filename'] for r in excluded]}")
        if len(coarse) == 0:
            print(f"  [警告] 召回结果为空！请检查：1)索引库是否有数据 2)查询文件是否是索引库中唯一零件")

        t1 = time.time()

        print(f"  耗时: {t1-t0:.2f}s")
        print(f"  {'#':<4} {'文件名':<35} {'相似度':<10}")
        print(f"  {'-'*55}")
        for i, r in enumerate(coarse[:10], 1):
            sim_type = "融合" if use_three_way else ("混合" if use_geometric else "嵌入")
            # 兼容不同检索模式的结果字段名
            filename = r.get('filename', Path(r.get('filepath', r.get('path', ''))).name)
            similarity = r.get('hybrid_similarity', r.get('final_score', r.get('similarity', 0)))
            print(f"  {i:<4} {filename:<35} {sim_type}:{similarity:.4f}")

        if not use_llm_rerank:
            for i, r in enumerate(coarse[:final_top]):
                r['rank'] = i + 1
            # 生成Word报告
            if save_report:
                report_gen = WordReportGenerator()
                report_file = report_gen.generate_report(
                    query_path, query_info, coarse[:final_top],
                    use_llm_rerank=False, save_path=report_path
                )
                print(f"\n[报告] Word报告已保存: {report_file}")
            return coarse[:final_top]

        # 阶段2: LLM 精排（候选数量根据粗筛结果动态调整）
        llm_rerank_candidates = min(30, len(coarse))  # 精排候选最多25个
        llm_input = coarse[:llm_rerank_candidates]

        # 检查是否使用视觉分析
        use_vision_rerank = False
        if use_vision and self.renderer:
            # 尝试自动渲染查询零件
            if query_views is None:
                print(f"\n[渲染] 正在检查查询零件视图...")
                # 先查找已有的视图
                query_views = find_view_images(query_path, self.view_dir)
                available_query_views = sum(1 for v in query_views if v)

                # 如果没有视图，尝试自动渲染
                if available_query_views < 3:
                    print(f"  [渲染] 找到 {available_query_views} 个现有视图，不足3个，尝试自动渲染...")
                    rendered_views = self.renderer.render_to_cache(query_path)
                    if rendered_views:
                        query_views = rendered_views
                        available_query_views = 6
                        print(f"  [渲染] 自动渲染完成: 6个视图")
                    else:
                        print(f"  [渲染] 自动渲染失败，使用已有视图")
                else:
                    print(f"  [渲染] 使用已有视图: {available_query_views}/6")
            else:
                available_query_views = sum(1 for v in query_views if v)

            # 检查候选零件是否有视图
            candidates_with_views = 0
            candidate_view_list = []  # 存储每个候选零件的视图路径
            if self.view_dir or self.renderer:
                for c in llm_input:
                    cand_views = find_view_images(c.get('path', ''), self.view_dir)
                    available_cand = sum(1 for v in cand_views if v)

                    # 如果候选零件视图不足，尝试渲染
                    if available_cand < 3 and self.renderer:
                        cand_name = Path(c['filename']).stem
                        print(f"  [渲染] 候选 {cand_name} 视图不足，尝试自动渲染...")
                        rendered = self.renderer.render_to_cache(c.get('path', ''))
                        if rendered:
                            cand_views = rendered
                            available_cand = 6

                    if available_cand >= 3:
                        candidates_with_views += 1

                    candidate_view_list.append(cand_views)

            if available_query_views >= 3 and candidates_with_views > 0:
                use_vision_rerank = True
                print(f"\n[阶段2] 多模态视觉LLM精排 {len(llm_input)} 个候选...")
                print(f"  查询零件视图: {available_query_views}/6")
                print(f"  候选零件视图: {candidates_with_views}/{len(llm_input)}")
            else:
                print(f"\n[阶段2] 文本LLM精排 {len(llm_input)} 个候选...")
                if available_query_views < 3:
                    print(f"  注: 查询零件视图不足（{available_query_views}/6），使用文本模式")
                elif candidates_with_views == 0:
                    print(f"  注: 候选零件无足够视图，使用文本模式")
        else:
            print(f"\n[阶段2] 文本LLM精排 {len(llm_input)} 个候选...")

        t2 = time.time()

        query_desc = generate_description(query_info)

        # 生成STP文本特征（用于特征提取增强）
        query_stp_features = f"""面数: {query_info['num_faces']}
边数: {query_info['num_edges']}
顶点数: {query_info['num_vertices']}
尺寸: {query_info['bbox_dims']}
面类型: {query_info['face_types']}
边类型: {query_info['edge_types']}
半径列表: {query_info['unique_radii']}"""

        # 选择精排方法
        # 3D视觉增强：渲染旋转动画GIF
        query_rotation_gif = None
        candidate_rotation_gifs = None

        if use_vision_rerank and use_3d_vision and self.renderer:
            print(f"\n[3D Vision] 正在渲染旋转动画...")
            # 渲染查询零件旋转动画
            query_rotation_gif = self.renderer.render_rotation_gif(query_path)
            if query_rotation_gif:
                print(f"  [3D Vision] 查询零件旋转动画: {Path(query_rotation_gif).name}")
            else:
                print(f"  [3D Vision] 查询零件旋转动画渲染失败")

            # 渲染候选零件旋转动画
            candidate_rotation_gifs = []
            for c in llm_input:
                cand_path = c.get('path', '')
                if cand_path:
                    cand_gif = self.renderer.render_rotation_gif(cand_path)
                    candidate_rotation_gifs.append(cand_gif)
                    if cand_gif:
                        print(f"  [3D Vision] 候选 {Path(cand_path).stem} 旋转动画完成")
                else:
                    candidate_rotation_gifs.append(None)

        if use_vision_rerank and use_feature_extraction:
            # 使用特征增强的多模态精排
            print(f"  使用特征增强精排模式")
            ranked = self.reranker.rerank_with_feature_extraction(
                query_path, query_desc, llm_input, top_k=final_top,
                query_views=query_views,
                query_stp_features=query_stp_features,
                candidate_views=candidate_view_list,
                renderer=self.renderer,
                reference_images=reference_images,
                reference_dimensions=reference_dimensions,
                custom_query_text=custom_query_text
            )
        elif use_vision_rerank:
            # 使用普通多模态精排
            ranked = self.reranker.rerank_with_vision(
                query_path, query_desc, llm_input, top_k=final_top,
                query_views=query_views,
                reference_images=reference_images,
                reference_dimensions=reference_dimensions,
                custom_query_text=custom_query_text,
                query_rotation_gif=query_rotation_gif,
                candidate_rotation_gifs=candidate_rotation_gifs
            )
        else:
            # 使用纯文本精排
            ranked = self.reranker.rerank(
                query_desc, llm_input, top_k=final_top,
                reference_images=reference_images,
                reference_dimensions=reference_dimensions,
                custom_query_text=custom_query_text
            )

        t3 = time.time()

        print(f"  耗时: {t3-t2:.2f}s")
        print(f"\n{'='*60}")
        print(f"最终结果")
        print(f"{'='*60}")
        for i, r in enumerate(ranked, 1):
            emb_sim = r.get('embedding_similarity', 0)
            geo_sim = r.get('geometric_similarity', {})
            geo_overall = geo_sim.get('overall', 0) if geo_sim else 0
            hybrid_sim = r.get('hybrid_similarity', 0)
            has_views = r.get('has_views', 0)
            feature_match = r.get('feature_match_score', 0)
            visual_sim = r.get('visual_similarity', 0)  # 视觉相似度评分

            print(f"  #{i} {r['filename']}")
            score_label = "检索回退评分" if r.get('llm_fallback') else "LLM评分"
            print(f"     {score_label}: {r['similarity_score']:.2f}")
            print(f"     Embedding: {emb_sim:.4f}  几何: {geo_overall:.4f}  混合: {hybrid_sim:.4f}")
            if use_vision_rerank:
                vision_info = f"视图: {has_views}个"
                if use_3d_vision and query_rotation_gif:
                    vision_info += " + 3D旋转动画"
                # 显示视觉相似度评分
                if isinstance(visual_sim, (int, float)) and visual_sim > 0:
                    print(f"     {vision_info}  视觉相似度: {visual_sim:.4f}  视觉匹配: {r.get('visual_match', 'N/A')}")
                else:
                    print(f"     {vision_info}  视觉匹配: {r.get('visual_match', 'N/A')}")
            if isinstance(feature_match, (int, float)) and feature_match > 0:
                print(f"     特征匹配: {feature_match:.4f}  关键匹配: {r.get('key_matches', [])}")
            if r.get('visual_features'):
                keywords = r['visual_features'].get('visual_keywords', [])
                print(f"     视觉关键词: {keywords}")
            print(f"     理由: {r['reason']}")

        # 生成Word报告
        if save_report:
            report_gen = WordReportGenerator()
            # 构建候选GIF字典 {filename: gif_path}
            candidate_gif_dict = {}
            if candidate_rotation_gifs and len(candidate_rotation_gifs) > 0:
                for i, gif_path in enumerate(candidate_rotation_gifs):
                    if gif_path and i < len(ranked):
                        filename = ranked[i].get('filename', f'候选{i+1}')
                        candidate_gif_dict[filename] = gif_path

            report_file = report_gen.generate_report(
                query_path, query_info, ranked,
                use_llm_rerank=True, save_path=report_path,
                query_rotation_gif=query_rotation_gif,
                candidate_rotation_gifs=candidate_gif_dict
            )
            print(f"\n[报告] Word报告已保存: {report_file}")

        if debugger:
            debugger.print_report()

        return ranked


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STP 零件相似度检索 (大模型版 - 几何特征增强 + 多模态视觉分析)")
    parser.add_argument("action", choices=["build", "search", "search-fast"])
    parser.add_argument("--dir", default="./parts")
    parser.add_argument("--query")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--coarse", type=int, default=20)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default=None,
                        help="Embedding API base URL，例如阿里云兼容接口")
    parser.add_argument("--llm-model", default="qwen2.5-72b-instruct")
    parser.add_argument("--llm-api-key", default=None,
                        help="LLM专用API密钥；本地OpenAI兼容服务通常可填EMPTY，默认沿用--api-key")
    parser.add_argument("--llm-base-url", default=None,
                        help="LLM专用OpenAI兼容地址，例如 http://34服务器IP:18223/v1；默认沿用--base-url")
    parser.add_argument("--llm-timeout", type=float, default=180.0,
                        help="LLM单次请求超时秒数（默认180）")
    parser.add_argument("--llm-max-retries", type=int, default=1,
                        help="LLM请求自动重试次数（默认1）")
    parser.add_argument("--llm-response-format", choices=["auto", "json", "none"], default="auto",
                        help="JSON响应参数兼容模式：auto不支持时自动去参重试；json强制；none不发送（默认auto）")
    parser.add_argument("--llm-thinking", choices=["auto", "on", "off"], default="auto",
                        help="Qwen思考模式：Qwen3.5本地服务建议设为off，避免content为空（默认auto）")
    parser.add_argument("--embed-model", default="text-embedding-v3")
    parser.add_argument("--no-geometric", action="store_true",
                        help="禁用几何特征相似度计算")
    parser.add_argument("--no-report", action="store_true",
                        help="禁用生成Word报告")
    parser.add_argument("--report-path", default=None,
                        help="指定Word报告保存路径")
    parser.add_argument("--view-dir", default=None,
                        help="视图图片根目录（用于多模态视觉分析）")
    parser.add_argument("--query-views", nargs='+', default=None,
                        help="查询零件的显式视图路径，建议按front top left right bottom back顺序提供至少3张")
    parser.add_argument("--no-vision", action="store_true",
                        help="禁用视觉分析，仅使用文本LLM精排")
    parser.add_argument("--no-feature-extraction", action="store_true",
                        help="禁用特征提取增强，直接使用多模态对比")
    parser.add_argument("--reference-images", nargs='+', default=None,
                        help="参考图片路径列表（如1.png 2.jpg），用于指定目标零件样式/尺寸")
    parser.add_argument("--reference-dimensions", default=None,
                        help="参考尺寸，格式为'长度x宽度x高度'（如100x60x30），单位mm")
    parser.add_argument("--custom-query", default=None,
                        help="自定义查找条件文本，用于定义查找逻辑（如'查找U形零件，尺寸100x60x30mm，有2个孔'）")
    parser.add_argument("--db-path", default="./chroma_parts",
                        help="向量数据库路径（默认: ./chroma_parts）")
    parser.add_argument("--use-faiss", action="store_true",
                        help="使用FAISS索引（后端索引格式）")
    parser.add_argument("--use-multi-vector", action="store_true",
                        help="使用多向量融合检索（文本+几何+视觉三路向量），提高召回率")
    parser.add_argument("--use-three-way", action="store_true",
                        help="使用三路融合检索（文本+几何+视觉独立召回+融合排序），提高召回质量")
    parser.add_argument("--text-recall", type=int, default=150,
                        help="文本召回数量（默认150）")
    parser.add_argument("--geo-recall", type=int, default=300,
                        help="几何召回数量（默认300）")
    parser.add_argument("--visual-recall", type=int, default=150,
                        help="视觉召回数量（默认150）")
    parser.add_argument("--fusion-top", type=int, default=30,
                        help="融合排序后保留数量（默认30）")
    parser.add_argument("--use-3d-vision", action="store_true",
                        help="在精筛中使用三维旋转动画（GIF），增强三维形态理解")
    parser.add_argument("--debug-recall", default=None,
                        help="调试模式：追踪指定目标零件文件名在各阶段的召回状态（如'5158F-QJ753-00002.stp'）")
    args = parser.parse_args()

    # 解析参考尺寸
    reference_dimensions = None
    if args.reference_dimensions:
        try:
            dims = args.reference_dimensions.split('x')
            if len(dims) == 3:
                reference_dimensions = {
                    'length': float(dims[0]),
                    'width': float(dims[1]),
                    'height': float(dims[2])
                }
                print(f"参考尺寸: {reference_dimensions['length']}x{reference_dimensions['width']}x{reference_dimensions['height']}mm")
        except Exception as e:
            print(f"警告: 无法解析参考尺寸 '{args.reference_dimensions}'，格式应为'长度x宽度x高度'")

    if args.custom_query:
        print(f"自定义查找条件: {args.custom_query}")

    # 当显式指定LLM服务器但未提供密钥时，给本地OpenAI兼容服务传入占位值。
    # OpenAI Python客户端要求api_key非空，即使服务端本身不校验密钥。
    resolved_llm_api_key = args.llm_api_key
    if args.llm_base_url and not resolved_llm_api_key:
        resolved_llm_api_key = "EMPTY"

    engine = STPSearchEngine(
        api_key=args.api_key,
        base_url=args.base_url,
        embedding_model=args.embed_model,
        llm_model=args.llm_model,
        llm_api_key=resolved_llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_response_format=args.llm_response_format,
        llm_thinking=args.llm_thinking,
        db_path=args.db_path,
        use_faiss=args.use_faiss,
        view_dir=args.view_dir,
    )

    if args.action == "build":
        engine.build_index(args.dir, view_dir=args.view_dir)

    elif args.action == "search":
        if not args.query:
            print("请指定 --query")
            exit(1)
        engine.search(args.query, coarse_top=args.coarse,
                      final_top=args.top, use_llm_rerank=True,
                      use_vision=not args.no_vision,
                      use_geometric=not args.no_geometric,
                      use_feature_extraction=not args.no_feature_extraction,
                      use_three_way=args.use_three_way,
                      use_3d_vision=args.use_3d_vision,
                      query_views=args.query_views,
                      save_report=not args.no_report,
                      report_path=args.report_path,
                      reference_images=args.reference_images,
                      reference_dimensions=reference_dimensions,
                      custom_query_text=args.custom_query,
                      debug_recall=args.debug_recall,
                      text_recall_k=args.text_recall,
                      geo_recall_k=args.geo_recall,
                      visual_recall_k=args.visual_recall,
                      fusion_top_k=args.fusion_top)

    elif args.action == "search-fast":
        if not args.query:
            print("请指定 --query")
            exit(1)
        engine.search(args.query, coarse_top=args.top,
                      final_top=args.top, use_llm_rerank=False,
                      use_vision=not args.no_vision,
                      use_geometric=not args.no_geometric,
                      use_feature_extraction=False,
                      use_3d_vision=False,
                      save_report=not args.no_report,
                      report_path=args.report_path,
                      reference_images=args.reference_images,
                      reference_dimensions=reference_dimensions,
                      custom_query_text=args.custom_query,
                      debug_recall=args.debug_recall)


# 使用说明（真实密钥请通过环境变量或安全的密钥管理方式提供）：
#
# 1. 34服务器文本测试（当前服务器HTTP服务异常，修复后使用）
# python stp.py search --query "C:\path\query.stp" --top 7 --api-key "$env:EMBEDDING_API_KEY" --base-url "https://embedding-service.example.com/v1" --embed-model "text-embedding-v3" --llm-api-key "EMPTY" --llm-base-url "http://10.100.0.35:8000/v1" --llm-model "Qwen3.5-35B-A3B"  --llm-thinking off --llm-response-format auto --use-three-way --no-vision --no-feature-extraction
#
# 2. 35服务器文本精排（已经过/models和纯文本聊天接口验证）
# python stp.py search --query "C:\Users\phillip\Desktop\project\teststp\测试件3.stp" --top 7 --api-key "$env:EMBEDDING_API_KEY" --base-url "--base-url "https://llm-8qgclixatgifoso3.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" --embed-model "text-embedding-v3" --llm-api-key "EMPTY" --llm-base-url "http://10.100.0.35:8000/v1" --llm-model "Qwen3.5-35B-A3B" --llm-thinking off --llm-response-format auto --use-three-way --no-vision --no-feature-extraction
#
# 3. 如果服务明确不接受response_format参数，在上述命令后增加：
# --llm-response-format none
