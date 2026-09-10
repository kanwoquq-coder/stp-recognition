# stp_ltr.py
"""
Learning to Rank (LTR) 精排模块
在现有多路召回基础上，通过少量人工标注数据训练精排模型，
替换/辅助当前的启发式融合排序，提升检索准确性和稳定性。

组件:
    - LTRFeatureExtractor: 从(query, candidate)对提取29维特征向量
    - LightGBMModel: LightGBM LambdaRank模型封装
    - MLPModel: sklearn MLPRegressor备用模型
    - LTRPipeline: 编排训练、评估、推理、标注流程
    - CLI入口: label/train/evaluate/feature-importance/predict
"""

import json
import math
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

import numpy as np

# 可选依赖
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 从当前项目导入
try:
    from stp_similarity import (
        parse_stp_deep, calculate_geometric_similarity,
        calculate_mfg_similarity, cosine_similarity,
        generate_description,
    )
except ImportError:
    # 相对导入
    import importlib
    spec = importlib.util.spec_from_file_location(
        "stp_similarity",
        os.path.join(os.path.dirname(__file__), "stp_similarity.py")
    )
    stp_sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stp_sim)
    parse_stp_deep = stp_sim.parse_stp_deep
    calculate_geometric_similarity = stp_sim.calculate_geometric_similarity
    calculate_mfg_similarity = stp_sim.calculate_mfg_similarity
    cosine_similarity = stp_sim.cosine_similarity
    generate_description = stp_sim.generate_description


# ============================================================
# 29维特征定义
# ============================================================

FEATURE_NAMES = [
    # 1. 文本相似度
    "text_similarity",
    # 2-9. 几何特征子评分 (8维)
    "geo_face_type_dist", "geo_topology", "geo_aspect_ratio",
    "geo_point_dist", "geo_complexity", "geo_feature_richness",
    "geo_compactness", "geo_rotational_symmetry",
    # 10-17. 制造特征子评分 (8维)
    "mfg_holes", "mfg_slots_pockets", "mfg_bosses", "mfg_ribs",
    "mfg_fillets_chamfers", "mfg_thread", "mfg_complexity",
    "mfg_vector_similarity",
    # 18. 视觉相似度
    "visual_similarity",
    # 19. 被几路召回命中
    "recall_count",
    # 20-22. 各路召回中的倒数排名
    "rank_inv_text", "rank_inv_geo", "rank_inv_visual",
    # 23-26. 面数/边数/体积/欧拉数差异
    "diff_face_count", "diff_edge_count", "diff_volume", "diff_euler",
    # 27. 圆柱面半径分布余弦相似度
    "cylinder_radius_sim",
    # 28. 是否有视觉特征
    "has_visual",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 29


def _parse_volume(info: dict) -> float:
    """从parse_stp_deep结果中获取体积估计值"""
    dims = info.get('bbox_dims', [0, 0, 0])
    return dims[0] * dims[1] * dims[2]


def _cylinder_radius_similarity(info1: dict, info2: dict) -> float:
    """
    计算两个零件的圆柱面半径分布余弦相似度
    基于cylinder_radii集合构建直方图
    """
    r1 = info1.get('cylinder_radii', [])
    r2 = info2.get('cylinder_radii', [])

    if not r1 or not r2:
        return 0.0

    # 构建半径直方图 (bin宽度=1mm，范围0-100mm)
    bins = np.arange(0, 100, 1.0)
    h1, _ = np.histogram(r1, bins=bins)
    h2, _ = np.histogram(r2, bins=bins)

    return cosine_similarity(h1.astype(np.float32), h2.astype(np.float32))


# ============================================================
# LTRFeatureExtractor
# ============================================================

class LTRFeatureExtractor:
    """
    从(query, candidate)对提取29维特征向量

    所有特征全部从现有计算结果中复用，无需新增解析开销。
    """

    def __init__(self, info_cache: Optional[Dict[str, dict]] = None):
        """
        Args:
            info_cache: 可选的全局信息缓存 {filepath: parse_stp_deep结果}
        """
        self._info_cache = info_cache or {}

    def extract(
        self,
        query_info: dict,
        candidate: dict,
        candidate_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        提取单个(query, candidate)对的29维特征向量

        Args:
            query_info: 查询件的parse_stp_deep结果
            candidate: 候选字典（含text_similarity, geo_similarity等字段）
            candidate_info: 候选件的parse_stp_deep结果（可选，用于diff特征）

        Returns:
            29维float32特征向量
        """
        vec = np.zeros(NUM_FEATURES, dtype=np.float32)

        # ========== 1. 文本相似度 ==========
        vec[0] = float(candidate.get('text_similarity', 0.5))

        # ========== 2-9. 几何特征子评分 (8维) ==========
        geo_sim = candidate.get('geometric_similarity', {})
        if isinstance(geo_sim, dict):
            geo_keys = [
                'face_type_distribution', 'topology', 'aspect_ratio',
                'point_distribution', 'complexity', 'feature_richness',
                'compactness', 'rotational_symmetry',
            ]
            for i, key in enumerate(geo_keys):
                vec[1 + i] = float(geo_sim.get(key, 0.5))

        # ========== 10-17. 制造特征子评分 (8维) ==========
        mfg_sim = {}
        if isinstance(geo_sim, dict) and 'manufacturing' in geo_sim:
            mfg_sim = geo_sim['manufacturing']
        elif isinstance(geo_sim, dict):
            mfg_sim = geo_sim  # 部分场景下直接使用
        # 尝试从candidate中直接获取mfg_similarity
        if not mfg_sim:
            mfg_sim = candidate.get('mfg_similarity', candidate.get('manufacturing_similarity', {}))

        mfg_keys = [
            'holes', 'slots_pockets', 'bosses', 'ribs',
            'fillets_chamfers', 'thread', 'complexity', 'vector_similarity',
        ]
        for i, key in enumerate(mfg_keys):
            vec[9 + i] = float(mfg_sim.get(key, 0.5)) if isinstance(mfg_sim, dict) else 0.5

        # ========== 18. 视觉相似度 ==========
        vec[17] = float(candidate.get('visual_similarity', 0.0))

        # ========== 19. 被几路召回命中 ==========
        recall_count = candidate.get('recall_count', 0)
        if not recall_count:
            # 从recall_sources推断
            recall_sources = candidate.get('recall_sources', [])
            recall_count = len(recall_sources) if isinstance(recall_sources, (list, tuple)) else 0
        vec[18] = float(min(recall_count, 3))

        # ========== 20-22. 各路召回中的倒数排名 ==========
        rrf_k = 60
        text_rank = candidate.get('text_rank', 0)
        vec[19] = 1.0 / (rrf_k + text_rank) if text_rank > 0 else 0.0

        geo_rank = candidate.get('geo_rank', 0)
        vec[20] = 1.0 / (rrf_k + geo_rank) if geo_rank > 0 else 0.0

        visual_rank = candidate.get('visual_rank', 0)
        vec[21] = 1.0 / (rrf_k + visual_rank) if visual_rank > 0 else 0.0

        # ========== 23-26. 差异特征 (4维) ==========
        if candidate_info is not None:
            # 面数差异
            q_faces = query_info.get('num_faces', 1)
            c_faces = candidate_info.get('num_faces', 1)
            vec[22] = 1.0 - abs(q_faces - c_faces) / max(q_faces, c_faces, 1)

            # 边数差异
            q_edges = query_info.get('num_edges', 1)
            c_edges = candidate_info.get('num_edges', 1)
            vec[23] = 1.0 - abs(q_edges - c_edges) / max(q_edges, c_edges, 1)

            # 体积差异
            q_vol = _parse_volume(query_info)
            c_vol = _parse_volume(candidate_info)
            max_vol = max(q_vol, c_vol, 1.0)
            vec[24] = 1.0 - abs(q_vol - c_vol) / max_vol

            # 欧拉数差异
            q_euler = query_info.get('euler', 0)
            c_euler = candidate_info.get('euler', 0)
            max_euler = max(abs(q_euler), abs(c_euler), 1)
            vec[25] = 1.0 - abs(q_euler - c_euler) / max_euler
        else:
            # 无可用的candidate_info时，使用默认值
            vec[22:26] = 0.5

        # ========== 27. 圆柱面半径分布相似度 ==========
        if candidate_info is not None:
            vec[26] = _cylinder_radius_similarity(query_info, candidate_info)
        else:
            vec[26] = 0.0

        # ========== 28. 是否有视觉特征 ==========
        has_visual = 0
        if candidate.get('visual_similarity', 0) > 0:
            has_visual = 1
        elif candidate.get('visual_rank', 0) > 0:
            has_visual = 1
        elif 'visual_similarity' in candidate:
            has_visual = 1
        vec[27] = float(has_visual)

        # 裁剪到[0,1]
        vec = np.clip(vec, 0.0, 1.0)

        return vec

    def extract_batch(
        self,
        query_info: dict,
        candidates: List[dict],
        info_cache: Optional[Dict[str, dict]] = None,
    ) -> np.ndarray:
        """
        批量提取特征向量

        Args:
            query_info: 查询件的parse_stp_deep结果
            candidates: 候选字典列表
            info_cache: 候选信息缓存 {filepath: info}

        Returns:
            (n_candidates, 29) 特征矩阵
        """
        cache = info_cache or self._info_cache
        features = []
        for cand in candidates:
            cand_info = None
            fp = cand.get('filepath', cand.get('path', ''))
            if fp and fp in cache:
                cand_info = cache[fp]
            features.append(self.extract(query_info, cand, cand_info))
        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """返回特征名称列表"""
        return list(FEATURE_NAMES)


# ============================================================
# LightGBMModel
# ============================================================

class LightGBMModel:
    """
    LightGBM LambdaRank模型封装

    使用LambdaRank直接优化NDCG，推理速度快（100个候选<10ms）。
    """

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        min_data_in_leaf: int = 5,
        num_iterations: int = 200,
        early_stopping_rounds: int = 20,
        seed: int = 42,
    ):
        """
        Args:
            num_leaves: 每棵树的最大叶子数
            learning_rate: 学习率
            min_data_in_leaf: 叶子节点最小数据量
            num_iterations: 最大迭代次数
            early_stopping_rounds: 早停轮数
            seed: 随机种子
        """
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'min_data_in_leaf': min_data_in_leaf,
            'seed': seed,
            'verbosity': -1,
            'boosting_type': 'gbdt',
        }
        self.num_iterations = num_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> dict:
        """
        训练LambdaRank模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 相关性标签 (n_samples,)，越高越相关
            group: 查询分组 (n_queries,)，每个查询的候选数
            eval_set: 可选的验证集 (X_val, y_val, group_val)

        Returns:
            训练历史信息
        """
        if not HAS_LIGHTGBM:
            raise RuntimeError(
                "LightGBM未安装。请执行: pip install lightgbm\n"
                "或使用MLPModel作为备选。"
            )

        train_data = lgb.Dataset(X, label=y, group=group)

        valid_sets = [train_data]
        valid_names = ['train']

        if eval_set is not None:
            X_val, y_val, group_val = eval_set
            valid_data = lgb.Dataset(X_val, label=y_val, group=group_val)
            valid_sets.append(valid_data)
            valid_names.append('eval')

        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.num_iterations,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds),
                       lgb.log_evaluation(period=0)],
        )
        self._is_trained = True

        history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
        }
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测排序分数

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            排序分数 (n_samples,)，越高越靠前
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("模型尚未训练，请先调用train()")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path: str) -> str:
        """
        保存模型到文件

        Args:
            path: 保存路径（如 model.txt）

        Returns:
            实际保存路径
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，无法保存")

        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        # 同时保存特征名称
        feat_path = path.replace('.txt', '_features.json')
        with open(feat_path, 'w', encoding='utf-8') as f:
            json.dump({'feature_names': FEATURE_NAMES, 'num_features': NUM_FEATURES}, f)
        return path

    def load(self, path: str) -> bool:
        """
        从文件加载模型

        Args:
            path: 模型文件路径

        Returns:
            是否加载成功
        """
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM未安装")

        if not os.path.exists(path):
            return False

        self.model = lgb.Booster(model_file=path)
        self._is_trained = True
        return True

    def feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性

        Returns:
            {特征名: 重要性分数} 字典
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")

        importance = self.model.feature_importance(importance_type='gain')
        return {FEATURE_NAMES[i]: float(v) for i, v in enumerate(importance)}


# ============================================================
# MLPModel (备用模型)
# ============================================================

class MLPModel:
    """
    sklearn MLPRegressor备用模型

    当LightGBM不可用时自动降级使用。
    使用pointwise方式训练（直接回归相关性分数）。
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32, 16),
        max_iter: int = 500,
        seed: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.seed = seed
        self.model: Optional[MLPRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: Optional[np.ndarray] = None,
    ) -> dict:
        """
        训练MLP回归模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 相关性标签 (n_samples,)
            group: 查询分组（MLP不使用，保留接口兼容）

        Returns:
            训练历史信息
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn未安装。请执行: pip install scikit-learn")

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
        self.model.fit(X_scaled, y)
        self._is_trained = True

        return {
            'loss': self.model.loss_,
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测相关性分数"""
        if not self._is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict(X_scaled)

    def save(self, path: str) -> str:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        import joblib
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

        # 保存特征名称
        feat_path = path.replace('.joblib', '_features.json')
        with open(feat_path, 'w', encoding='utf-8') as f:
            json.dump({'feature_names': FEATURE_NAMES, 'num_features': NUM_FEATURES}, f)
        return path

    def load(self, path: str) -> bool:
        """加载模型"""
        if not os.path.exists(path):
            return False

        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data.get('scaler')
        self._is_trained = True
        return True

    def feature_importance(self) -> Dict[str, float]:
        """MLP无原生特征重要性，使用绝对权重之和作为近似"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        # 使用第一层权重绝对值之和作为特征重要性近似
        coef = self.model.coefs_[0]
        importance = np.abs(coef).sum(axis=1)
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return {FEATURE_NAMES[i]: float(v) for i, v in enumerate(importance)}


# ============================================================
# 评估指标
# ============================================================

def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """
    计算NDCG@k

    Args:
        y_true: 真实相关性标签
        y_pred: 预测分数
        k: 截断数，None表示全部

    Returns:
        NDCG分数
    """
    if k:
        y_true = y_true[:k]
        y_pred = y_pred[:k]

    # 按预测分数降序排序
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]

    # DCG
    dcg = y_true_sorted[0]
    for i in range(1, len(y_true_sorted)):
        dcg += y_true_sorted[i] / np.log2(i + 1)

    # IDCG
    y_true_ideal = np.sort(y_true)[::-1]
    idcg = y_true_ideal[0]
    for i in range(1, len(y_true_ideal)):
        idcg += y_true_ideal[i] / np.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算Pairwise Accuracy
    正确排序的pair比例

    Returns:
        准确率 [0, 1]
    """
    correct = 0
    total = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                # 预测排序应与真实排序一致
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    correct += 1
    return correct / max(total, 1)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算Spearman等级相关系数

    Returns:
        相关系数 [-1, 1]
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


# ============================================================
# LTRPipeline
# ============================================================

class LTRPipeline:
    """
    学习排序(LTR)全流程编排

    支持:
    - 交互式数据标注
    - 特征提取与训练数据准备
    - 模型训练（LightGBM / MLP）
    - 评估（NDCG, Pairwise Accuracy, Spearman）
    - 推理排序
    - 特征重要性分析
    """

    def __init__(
        self,
        model_type: str = 'lightgbm',
        model_path: Optional[str] = None,
        info_cache: Optional[Dict[str, dict]] = None,
    ):
        """
        Args:
            model_type: 模型类型 'lightgbm' 或 'mlp'
            model_path: 预训练模型路径（可选）
            info_cache: 零件信息缓存
        """
        self.model_type = model_type
        self.feature_extractor = LTRFeatureExtractor(info_cache=info_cache)
        self.model = self._create_model()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _create_model(self):
        """根据model_type创建模型实例"""
        if self.model_type == 'lightgbm':
            if HAS_LIGHTGBM:
                return LightGBMModel()
            else:
                print("[LTR] LightGBM未安装，自动降级到MLP模型")
                self.model_type = 'mlp'
                return MLPModel()
        elif self.model_type == 'mlp':
            return MLPModel()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def prepare_training_data(
        self,
        annotation_path: str,
        query_infos: Optional[Dict[str, dict]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从标注数据加载并提取特征，构建训练数据

        Args:
            annotation_path: 标注JSON文件路径
            query_infos: 可选的查询件信息缓存 {query_path: info}

        Returns:
            (X, y, group): 特征矩阵、标签、查询分组
        """
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        X_list = []
        y_list = []
        group_list = []

        # 处理listwise标注
        for q in data.get('listwise', []):
            query_path = q['query_path']
            query_info = self._get_query_info(query_path, query_infos)

            candidates = q['candidates']
            group_size = 0

            for cand in candidates:
                cand_path = cand['path']
                cand_info = self._get_candidate_info(cand_path)

                # 构建候选字典（模拟检索结果格式）
                cand_dict = {
                    'filepath': cand_path,
                    'path': cand_path,
                    'text_similarity': cand.get('text_similarity', 0.5),
                    'visual_similarity': cand.get('visual_similarity', 0.0),
                    'recall_count': cand.get('recall_count', 1),
                    'recall_sources': cand.get('recall_sources', ['text']),
                    'text_rank': cand.get('text_rank', 1),
                    'geo_rank': cand.get('geo_rank', 0),
                    'visual_rank': cand.get('visual_rank', 0),
                }

                # 几何相似度
                if cand_info and query_info:
                    geo_sim = calculate_geometric_similarity(query_info, cand_info)
                    cand_dict['geometric_similarity'] = geo_sim
                    # 制造特征相似度
                    mfg1 = query_info.get('mfg_features')
                    mfg2 = cand_info.get('mfg_features')
                    if mfg1 and mfg2:
                        mfg_sim = calculate_mfg_similarity(mfg1, mfg2)
                        cand_dict['mfg_similarity'] = mfg_sim

                feat = self.feature_extractor.extract(query_info, cand_dict, cand_info)
                X_list.append(feat)
                y_list.append(float(cand['relevance']))
                group_size += 1

            if group_size > 0:
                group_list.append(group_size)

        # 处理pairwise标注（转换为listwise格式）
        pairs = data.get('pairs', [])
        pair_groups = {}
        for p in pairs:
            qpath = p['query_path']
            if qpath not in pair_groups:
                pair_groups[qpath] = {}
            for key in ['candidate_a', 'candidate_b']:
                cpath = p[key]
                if cpath not in pair_groups[qpath]:
                    # preference: 1.0表示a优于b, 0.0表示b优于a
                    # 转换为相关性: 优的给3, 差的给1, 中间给2
                    pref = p.get('preference', 0.5)
                    if key == 'candidate_a':
                        relevance = 3.0 if pref > 0.5 else 1.0
                    else:
                        relevance = 1.0 if pref > 0.5 else 3.0
                    pair_groups[qpath][cpath] = {'relevance': relevance}

        for qpath, cands in pair_groups.items():
            query_info = self._get_query_info(qpath, query_infos)
            group_size = 0
            for cpath, cinfo in cands.items():
                cand_info = self._get_candidate_info(cpath)
                cand_dict = {
                    'filepath': cpath, 'path': cpath,
                    'text_similarity': 0.5, 'visual_similarity': 0.0,
                    'recall_count': 1, 'recall_sources': ['text'],
                    'text_rank': 1, 'geo_rank': 0, 'visual_rank': 0,
                }
                if cand_info and query_info:
                    geo_sim = calculate_geometric_similarity(query_info, cand_info)
                    cand_dict['geometric_similarity'] = geo_sim
                feat = self.feature_extractor.extract(query_info, cand_dict, cand_info)
                X_list.append(feat)
                y_list.append(cinfo['relevance'])
                group_size += 1
            if group_size > 0:
                group_list.append(group_size)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        group = np.array(group_list, dtype=np.int32)

        print(f"[LTR] 训练数据: {len(X)} 样本, {len(group)} 查询组")
        print(f"[LTR] 标签范围: [{y.min():.1f}, {y.max():.1f}]")
        return X, y, group

    def train(
        self,
        annotation_path: str,
        eval_annotation_path: Optional[str] = None,
        query_infos: Optional[Dict[str, dict]] = None,
    ) -> dict:
        """
        训练LTR模型

        Args:
            annotation_path: 训练标注数据路径
            eval_annotation_path: 验证标注数据路径（可选，用于早停）
            query_infos: 查询件信息缓存

        Returns:
            训练历史信息
        """
        X, y, group = self.prepare_training_data(annotation_path, query_infos)

        eval_set = None
        if eval_annotation_path:
            X_val, y_val, group_val = self.prepare_training_data(
                eval_annotation_path, query_infos
            )
            eval_set = (X_val, y_val, group_val)

        # 训练
        print(f"[LTR] 开始训练 {self.model_type} 模型...")
        t0 = time.time()

        if isinstance(self.model, LightGBMModel):
            history = self.model.train(X, y, group, eval_set)
        else:
            history = self.model.train(X, y, group)

        elapsed = time.time() - t0
        print(f"[LTR] 训练完成, 耗时 {elapsed:.2f}s")

        return history

    def evaluate(
        self,
        annotation_path: str,
        query_infos: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            annotation_path: 标注数据路径
            query_infos: 查询件信息缓存

        Returns:
            评估指标字典
        """
        if not self.model.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train()")

        X, y, group = self.prepare_training_data(annotation_path, query_infos)

        # 预测
        y_pred = self.model.predict(X)

        # 按group计算指标
        ndcg1_list = []
        ndcg3_list = []
        ndcg5_list = []
        acc_list = []
        spear_list = []

        idx = 0
        for g in group:
            y_true_g = y[idx:idx + g]
            y_pred_g = y_pred[idx:idx + g]

            # 按预测排序
            order = np.argsort(y_pred_g)[::-1]
            y_true_sorted = y_true_g[order]

            ndcg1_list.append(ndcg_score(y_true_g, y_pred_g, k=min(1, g)))
            ndcg3_list.append(ndcg_score(y_true_g, y_pred_g, k=min(3, g)))
            ndcg5_list.append(ndcg_score(y_true_g, y_pred_g, k=min(5, g)))
            acc_list.append(pairwise_accuracy(y_true_g, y_pred_g))
            spear_list.append(spearman_correlation(y_true_g, y_pred_g))

            idx += g

        results = {
            'ndcg@1': float(np.mean(ndcg1_list)),
            'ndcg@3': float(np.mean(ndcg3_list)),
            'ndcg@5': float(np.mean(ndcg5_list)),
            'pairwise_accuracy': float(np.mean(acc_list)),
            'spearman': float(np.mean(spear_list)),
            'num_queries': len(group),
            'num_samples': len(y),
        }
        return results

    def rerank(
        self,
        query_info: dict,
        candidates: List[dict],
        info_cache: Optional[Dict[str, dict]] = None,
    ) -> List[dict]:
        """
        使用LTR模型对候选列表进行重排序

        Args:
            query_info: 查询件的parse_stp_deep结果
            candidates: 候选列表（需包含text_similarity等字段）
            info_cache: 候选信息缓存

        Returns:
            重排序后的候选列表（含ltr_score字段）
        """
        if not self.model.is_trained:
            raise RuntimeError("模型尚未训练，请先训练或加载模型")

        # 提取特征
        X = self.feature_extractor.extract_batch(query_info, candidates, info_cache)

        # 预测
        ltr_scores = self.model.predict(X)

        # 添加到候选
        for i, cand in enumerate(candidates):
            cand['ltr_score'] = float(ltr_scores[i])

        # 按LTR分数排序
        candidates.sort(key=lambda x: x.get('ltr_score', 0), reverse=True)

        return candidates

    def save_model(self, path: str) -> str:
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return self.model.save(path)

    def load_model(self, path: str) -> bool:
        """加载模型"""
        return self.model.load(path)

    def feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.model.feature_importance()

    def _get_query_info(self, path: str, cache: Optional[dict] = None) -> dict:
        """获取查询件信息"""
        if cache and path in cache:
            return cache[path]
        return parse_stp_deep(path)

    def _get_candidate_info(self, path: str) -> Optional[dict]:
        """获取候选件信息，优先从缓存读取"""
        if path in self.feature_extractor._info_cache:
            return self.feature_extractor._info_cache[path]
        try:
            info = parse_stp_deep(path)
            self.feature_extractor._info_cache[path] = info
            return info
        except Exception:
            return None


# ============================================================
# 交互式标注工具
# ============================================================

def interactive_label(
    index,
    query_path: str,
    candidates: List[dict],
    output_path: str,
    existing_data: Optional[dict] = None,
):
    """
    交互式标注：对检索结果打分0-5

    Args:
        index: 检索引擎实例（需有search方法）
        query_path: 查询STP文件路径
        candidates: 候选列表
        output_path: 标注数据保存路径
        existing_data: 已有的标注数据
    """
    data = existing_data or {'metadata': {}, 'pairs': [], 'listwise': []}

    print(f"\n{'='*60}")
    print(f"交互式标注: {Path(query_path).name}")
    print(f"共 {len(candidates)} 个候选")
    print(f"{'='*60}")

    query_name = Path(query_path).name
    scored_candidates = []

    for i, cand in enumerate(candidates):
        filename = cand.get('filename', Path(cand.get('filepath', cand.get('path', ''))).name)
        similarity = cand.get('hybrid_similarity', cand.get('final_score', cand.get('similarity', 0)))

        print(f"\n--- 候选 #{i+1}: {filename} ---")
        print(f"  相似度: {similarity:.4f}")

        # 显示几何特征
        geo_sim = cand.get('geometric_similarity', {})
        if isinstance(geo_sim, dict):
            print(f"  几何面类型: {geo_sim.get('face_type_distribution', 'N/A')}")
            print(f"  拓扑: {geo_sim.get('topology', 'N/A')}")

        # 显示制造特征
        mfg = cand.get('mfg_similarity', {})
        if isinstance(mfg, dict):
            print(f"  孔特征: {mfg.get('holes', 'N/A')}")

        # 用户打分
        while True:
            try:
                score = input(f"  请输入相关性评分 (0=无关, 5=完全相同, q=跳过): ").strip()
                if score.lower() == 'q':
                    break
                score = int(score)
                if 0 <= score <= 5:
                    scored_candidates.append({
                        'path': cand.get('filepath', cand.get('path', '')),
                        'relevance': score,
                        'text_similarity': cand.get('text_similarity', 0.5),
                        'visual_similarity': cand.get('visual_similarity', 0.0),
                        'recall_count': cand.get('recall_count', 1),
                        'text_rank': cand.get('text_rank', 1),
                        'geo_rank': cand.get('geo_rank', 0),
                        'visual_rank': cand.get('visual_rank', 0),
                        'recall_sources': cand.get('recall_sources', ['text']),
                    })
                    break
                else:
                    print("  请输入0-5之间的整数")
            except ValueError:
                print("  请输入有效数字")

    # 添加到listwise
    if scored_candidates:
        entry = {
            'query_path': query_path,
            'candidates': scored_candidates,
        }
        data['listwise'].append(entry)

        # 自动生成pairwise偏好对
        for i in range(len(scored_candidates)):
            for j in range(i + 1, len(scored_candidates)):
                if scored_candidates[i]['relevance'] != scored_candidates[j]['relevance']:
                    pref = 1.0 if scored_candidates[i]['relevance'] > scored_candidates[j]['relevance'] else 0.0
                    data['pairs'].append({
                        'query_path': query_path,
                        'candidate_a': scored_candidates[i]['path'],
                        'candidate_b': scored_candidates[j]['path'],
                        'preference': pref,
                    })

    # 更新元数据
    data['metadata'] = {
        'created': data['metadata'].get('created', datetime.now().isoformat()),
        'updated': datetime.now().isoformat(),
        'num_queries': len(data['listwise']),
        'num_pairs': len(data['pairs']),
    }

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n标注数据已保存: {output_path}")
    print(f"  共 {len(data['listwise'])} 个查询, {len(data['pairs'])} 个pair")

    return data


# ============================================================
# CLI入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LTR学习排序模块 - 训练/评估/标注/推理"
    )
    parser.add_argument("action", choices=[
        "label", "train", "evaluate", "feature-importance", "predict"
    ])
    parser.add_argument("--annotation", default="ltr_annotation.json",
                        help="标注数据路径")
    parser.add_argument("--eval-annotation", default=None,
                        help="验证标注数据路径（用于早停）")
    parser.add_argument("--model", default="ltr_model.txt",
                        help="模型路径 (LightGBM: .txt, MLP: .joblib)")
    parser.add_argument("--model-type", choices=["lightgbm", "mlp"],
                        default="lightgbm", help="模型类型")
    parser.add_argument("--query", help="查询STP文件路径（用于标注/推理）")
    parser.add_argument("--candidates", default=None,
                        help="候选JSON文件路径（用于推理）")
    parser.add_argument("--output", default=None,
                        help="输出路径")
    parser.add_argument("--info-cache", default=None,
                        help="信息缓存JSON文件路径")

    # 训练参数
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-iterations", type=int, default=200)

    args = parser.parse_args()

    # 加载信息缓存
    info_cache = {}
    if args.info_cache and os.path.exists(args.info_cache):
        with open(args.info_cache, 'r', encoding='utf-8') as f:
            info_cache = json.load(f)

    pipeline = LTRPipeline(
        model_type=args.model_type,
        model_path=args.model if args.action in ["evaluate", "predict", "feature-importance"] else None,
        info_cache=info_cache,
    )

    # ----- 标注 -----
    if args.action == "label":
        if not args.query:
            print("请指定 --query")
            sys.exit(1)

        # 从stp_similarity导入检索引擎进行搜索
        sys.path.insert(0, os.path.dirname(__file__))
        from stp_similarity import STPSearchEngine, EmbeddingIndex

        # 使用现有的配置
        api_key = os.environ.get("API_KEY", "")
        base_url = os.environ.get("BASE_URL", None)
        embed_model = os.environ.get("EMBED_MODEL", "text-embedding-v3")
        if not api_key:
            print("请设置环境变量 API_KEY")
            sys.exit(1)

        # 加载现有标注数据
        existing_data = None
        if os.path.exists(args.annotation):
            try:
                with open(args.annotation, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"加载已有标注数据: {len(existing_data.get('listwise', []))} 个查询")
            except Exception:
                pass

        # 搜索并标注
        engine = STPSearchEngine(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embed_model,
        )
        results = engine.search(
            args.query,
            coarse_top=30,
            final_top=10,
            use_llm_rerank=False,
            use_geometric=True,
            use_three_way=True,
            save_report=False,
        )

        interactive_label(
            engine.index,
            args.query,
            results,
            args.annotation,
            existing_data,
        )

    # ----- 训练 -----
    elif args.action == "train":
        if not os.path.exists(args.annotation):
            print(f"标注数据不存在: {args.annotation}")
            sys.exit(1)

        # 更新模型参数
        if isinstance(pipeline.model, LightGBMModel):
            pipeline.model.params['num_leaves'] = args.num_leaves
            pipeline.model.params['learning_rate'] = args.learning_rate
            pipeline.model.params['num_iterations'] = args.num_iterations
        elif isinstance(pipeline.model, MLPModel):
            pass  # MLP使用默认参数

        pipeline.train(args.annotation, args.eval_annotation)

        # 保存模型
        pipeline.save_model(args.model)

        # 评估
        results = pipeline.evaluate(args.annotation)
        print(f"\n[LTR] 训练集评估结果:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        if args.eval_annotation and os.path.exists(args.eval_annotation):
            eval_results = pipeline.evaluate(args.eval_annotation)
            print(f"\n[LTR] 验证集评估结果:")
            for k, v in eval_results.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ----- 评估 -----
    elif args.action == "evaluate":
        if not os.path.exists(args.annotation):
            print(f"标注数据不存在: {args.annotation}")
            sys.exit(1)

        results = pipeline.evaluate(args.annotation)
        print(f"\n[LTR] 评估结果:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ----- 特征重要性 -----
    elif args.action == "feature-importance":
        if not pipeline.model.is_trained:
            print("模型未训练或未加载")
            sys.exit(1)

        importance = pipeline.feature_importance()
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n[LTR] 特征重要性 (共{len(sorted_imp)}个特征):")
        print(f"  {'#':<4} {'特征名':<30} {'重要性':<12}")
        print(f"  {'-'*46}")
        for i, (name, val) in enumerate(sorted_imp, 1):
            bar = '█' * int(val * 50)
            print(f"  {i:<4} {name:<30} {val:.4f}  {bar}")

    # ----- 推理 -----
    elif args.action == "predict":
        if not pipeline.model.is_trained:
            print("模型未训练或未加载")
            sys.exit(1)
        if not args.query:
            print("请指定 --query")
            sys.exit(1)

        # 解析查询件
        query_info = parse_stp_deep(args.query)

        # 加载候选列表
        candidates = []
        if args.candidates and os.path.exists(args.candidates):
            with open(args.candidates, 'r', encoding='utf-8') as f:
                candidates = json.load(f)
            print(f"加载 {len(candidates)} 个候选")
        else:
            print("请指定 --candidates (候选JSON文件路径)")
            sys.exit(1)

        # 补充candidate_info
        for cand in candidates:
            fp = cand.get('filepath', cand.get('path', ''))
            if fp and fp not in info_cache:
                try:
                    info_cache[fp] = parse_stp_deep(fp)
                except Exception:
                    pass

            # 补全几何相似度
            if fp in info_cache and query_info:
                geo_sim = calculate_geometric_similarity(query_info, info_cache[fp])
                cand['geometric_similarity'] = geo_sim

        # 重排序
        reranked = pipeline.rerank(query_info, candidates, info_cache)

        # 输出
        output_path = args.output or "ltr_predictions.json"
        output = []
        for i, cand in enumerate(reranked, 1):
            output.append({
                'rank': i,
                'filename': cand.get('filename', Path(cand.get('filepath', cand.get('path', ''))).name),
                'filepath': cand.get('filepath', cand.get('path', '')),
                'ltr_score': cand.get('ltr_score', 0),
                'original_score': cand.get('hybrid_similarity', cand.get('final_score', cand.get('similarity', 0))),
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n[LTR] 推理结果 ({len(output)} 个候选):")
        print(f"  {'#':<4} {'文件名':<35} {'LTR分数':<10} {'原始分数':<10}")
        print(f"  {'-'*59}")
        for r in output:
            print(f"  {r['rank']:<4} {r['filename']:<35} {r['ltr_score']:.4f}  {r['original_score']:.4f}")

        print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()