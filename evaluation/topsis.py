# evaluation/topsis.py
# TOPSIS 综合评价）
# 实现多指标综合评价，确定 “最小生物多样性阈值”：

# 导入项目根目录到sys.path，确保可以导入模块（快速调试用）
import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（21-A-final/）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path
sys.path.append(root_dir)

import numpy as np
from evaluation.diversity_index import DiversityEvaluator

class TOPSISEvaluator:
    """
    TOPSIS综合评价器（上层核心）

    实现理论模型7.2节：多指标（DEI、S、I、G）综合评价，确定最小生物多样性

    依赖：DiversityEvaluator的评估指标
    """
    def __init__(self, metrics_list: list[dict], weights: list = None):
        """
        :param metrics_list: 不同生物多样性水平下的指标列表 [{"DEI":..., "stability":..., ...}, ...]
        :param weights: 指标权重 [w_DEI, w_diversity, w_stability, w_synergy]
        """
        self.metrics_list = metrics_list
        # 默认权重（分解效率优先，其次稳定性、多样性、协同）
        self.weights = np.array(weights) if weights else np.array([0.4, 0.2, 0.25, 0.15])
        # 提取指标矩阵
        self.metrics_matrix = self._build_metrics_matrix()

    def _build_metrics_matrix(self) -> np.ndarray:
        """构建指标矩阵：n_samples × n_metrics"""
        metrics_names = ["DEI", "diversity_index", "stability", "synergy_gain"]
        matrix = []
        for metrics in self.metrics_list:
            row = [metrics[name] for name in metrics_names]
            matrix.append(row)
        return np.array(matrix)

    def _normalize_matrix(self) -> np.ndarray:
        """归一化指标矩阵（线性归一化）"""
        # 所有指标均为正向（越大越好）
        max_vals = np.max(self.metrics_matrix, axis=0)
        min_vals = np.min(self.metrics_matrix, axis=0)
        diff = max_vals - min_vals
        diff[diff == 0] = 1e-8  # 避免除零
        normalized = (self.metrics_matrix - min_vals) / diff
        return normalized

    def _weighted_matrix(self) -> np.ndarray:
        """加权归一化矩阵"""
        normalized = self._normalize_matrix()
        return normalized * self.weights

    def evaluate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行TOPSIS评价

        :return: (贴近度, 正理想解距离, 负理想解距离)
        """
        weighted = self._weighted_matrix()
        
        # 正理想解（各指标最大值）、负理想解（各指标最小值）
        pos_ideal = np.max(weighted, axis=0)
        neg_ideal = np.min(weighted, axis=0)
        
        # 计算距离
        pos_dist = np.sqrt(np.sum((weighted - pos_ideal)**2, axis=1))
        neg_dist = np.sqrt(np.sum((weighted - neg_ideal)**2, axis=1))
        
        # 贴近度（越接近1越好）
        closeness = neg_dist / (pos_dist + neg_dist)
        return closeness, pos_dist, neg_dist

    def find_min_diversity_threshold(self, diversity_series: np.ndarray, threshold_closeness: float = 0.8) -> float:
        """
        找到最小生物多样性阈值（贴近度≥threshold_closeness的最小多样性）
        
        :param diversity_series: 对应metrics_list的多样性值序列
        :param threshold_closeness: 可接受的最小贴近度
        :return: 最小生物多样性阈值
        """
        closeness, _, _ = self.evaluate()
        # 筛选贴近度≥阈值的样本
        valid_idx = np.where(closeness >= threshold_closeness)[0]
        if len(valid_idx) == 0:
            return np.min(diversity_series)  # 无满足条件的，返回最小值
        
        valid_diversity = diversity_series[valid_idx]
        return np.min(valid_diversity)

# 单元测试
if __name__ == "__main__":
    # 模拟不同生物多样性水平的指标数据
    metrics_list = [
        {"DEI": 0.02, "diversity_index": 0.1, "stability": 0.5, "synergy_gain": 0.05},  # 低多样性
        {"DEI": 0.04, "diversity_index": 0.3, "stability": 0.7, "synergy_gain": 0.1},   # 中低多样性
        {"DEI": 0.05, "diversity_index": 0.5, "stability": 0.8, "synergy_gain": 0.15},  # 中多样性
        {"DEI": 0.052, "diversity_index": 0.7, "stability": 0.82, "synergy_gain": 0.18}, # 中高多样性
        {"DEI": 0.053, "diversity_index": 0.9, "stability": 0.83, "synergy_gain": 0.19}  # 高多样性
    ]
    diversity_series = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # 初始化TOPSIS评估器
    topsis = TOPSISEvaluator(metrics_list, weights=[0.4, 0.2, 0.25, 0.15])
    closeness, pos_dist, neg_dist = topsis.evaluate()
    min_diversity = topsis.find_min_diversity_threshold(diversity_series, threshold_closeness=0.8)

    # 输出结果
    print("\n=== TOPSIS综合评价测试结果 ===")
    for i, (div, close) in enumerate(zip(diversity_series, closeness)):
        print(f"多样性{div:.1f} | 贴近度：{close:.4f}")
    print(f"\n最小生物多样性阈值（贴近度≥0.8）：{min_diversity:.1f}")