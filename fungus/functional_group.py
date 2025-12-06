# fungus/functional_group.py
# 真菌功能群分类

# 导入项目根目录到sys.path，确保可以导入data模块（快速调试用）
import sys
import os

# 获取当前脚本（data_loader.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（21-A-final/，即current_dir的上级目录）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path
sys.path.append(root_dir)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data.data_loader import FungalDataLoader

class FunctionalGroupClassifier:
    """
    真菌功能群分类器（中层核心）
    功能：基于特征空间聚类，划分F型（快速分解）、S型（慢速分解）、I型（中间型）
    依赖：底层data_loader的特征空间、sklearn KMeans
    """
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters  # 默认3类（F/S/I）
        self.kmeans_model = None      # KMeans模型
        self.cluster_labels = None    # 聚类标签（每个真菌的类别）
        self.cluster_centers = None   # 聚类中心
        self.functional_groups = None # 最终功能群标签（F/S/I）
        self.fungus_ids = None        # 真菌ID映射
        self.group_to_cluster = None  # 功能群到聚类标签映射

    def _elbow_method(self, feature_space: np.ndarray, max_k: int = 8) -> int:
        """
        肘部法则确定最优聚类数
        
        :param self: 类实例本身
        :param feature_space: 标准化特征空间(n_fungus x n_features)
        :type feature_space: np.ndarray
        :param max_k: 最大聚类数，默认8
        :type max_k: int
        :return: 最优聚类数k
        :rtype: int
        """
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(feature_space)
            inertias.append(kmeans.inertia_)
        
        # 计算肘部点（简单方法：寻找惯性下降最快的点）
        deltas = np.diff(inertias)
        delta_ratios = deltas[1:] / deltas[:-1]
        optimal_k = np.argmin(delta_ratios) + 2  # +2因为diff减少了两个索引
        print(f"✅ 肘部法则推荐最优聚类数：{optimal_k}（理论值3）")
        return optimal_k
    
    def _assign_groups_by_centers(self) -> None:
        """
        根据聚类中心划分功能群标签（F/S/I）
        
        :param self: 类实例本身
        """
        if self.cluster_centers is None or self.cluster_labels is None:
            raise ValueError("请先拟合KMeans模型以获取聚类中心")
        
        # 获取聚类中心在核心特征上的值
        # p_22（分解率，索引2）、mu_22（菌丝延伸率，索引5）
        center_features = []
        for i, center in enumerate(self.cluster_centers):
            p_22 = center[2]        # 分解率
            mu_22 = center[5]       # 菌丝延伸率
            center_features.append((i, p_22, mu_22))

        # 基于p_22和mu_22的加权和排序
        # 快速分解型(F)：高p_22、高mu_22
        center_scores = []
        for i, p_22, mu_22 in center_features:
            score = 0.6 * p_22 + 0.4 * mu_22 # 分解率权重更大
            center_scores.append((i, score, p_22, mu_22))
        
        # 按分数排序
        center_scores.sort(key=lambda x: x[1], reverse=True)

        # 分配功能群标签: 得分最高的为F型，最低为S型，中间为I型
        self.group_to_cluster = {}
        functional_labels = ['F', 'I', 'S']

        for idx, (cluster_id, score, p_22, mu_22) in enumerate(center_scores):
            if idx < len(functional_labels):
                group = functional_labels[idx]
                self.group_to_cluster[group] = cluster_id
                print(f"聚类中心 {cluster_id} 分配为功能群 '{group}' (得分：{score:.3f}, p_22：{p_22:.3f}, mu_22：{mu_22:.3f})")
        
        # 根据映射生成每个真菌的功能群标签
        self.functional_groups = []
        for label in self.cluster_labels:
            # 查找对应功能群
            for group, cluster_id in self.group_to_cluster.items():
                if label == cluster_id:
                    self.functional_groups.append(group)
                    break
            else:
                # 未找到对应功能群，标记为I型
                self.functional_groups.append('I')
        

    def fit(self, feature_space: np.ndarray, fungus_ids: np.ndarray = None) -> None:
        """
        拟合聚类模型，生成功能群标签
        
        :param self: 类实例本身
        :param feature_space: 标准化特征空间（来自FungalDataLoader）
        :type feature_space: np.ndarray
        :param fungus_ids: 真菌ID(可选，用于映射)
        :type fungus_ids: np.ndarray
        """
        # 验证特征空间
        if feature_space.ndim != 2:
            raise ValueError("特征空间应为二维数组 (n_fungus x n_features)")
        
        # # 使用肘部法则确定最优聚类数（可选）
        # optimal_k = self._elbow_method(feature_space, max_k=8)
        # self.n_clusters = optimal_k

        # 拟合KMeans模型
        print(f"✅ 开始K-means聚类，聚类数: {self.n_clusters}")
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.kmeans_model.fit_predict(feature_space)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        self.fungus_ids = fungus_ids if fungus_ids is not None else np.arange(len(feature_space))

        # 划分F/S/I型（基于理论模型的核心特征阈值）
        print("✅ 基于聚类中心分配功能群标签:")
        self._assign_groups_by_centers()

        # 输出分类统计
        group_counts = {g: self.functional_groups.count(g) for g in ['F', 'S', 'I']}
        print(f"✅ 功能群分类完成：{group_counts}")

    def get_group_traits(self, group: str) -> np.ndarray:
        """
        获取指定功能群的特征空间
        
        :param self: 类实例本身
        :param group: 功能群标签（'F'/'S'/'I'）
        :type group: str
        :return: 指定功能群的特征空间
        :rtype: np.ndarray
        """
        if self.functional_groups is None:
            raise ValueError("请先调用fit方法进行聚类拟合")
        if group not in ['F', 'S', 'I']:
            raise ValueError("功能群标签应为'F'、'S'或'I'")
        
        # 检查该功能群是否有数据
        if self.functional_groups.count(group) == 0:
            print(f"⚠️  警告：功能群'{group}'没有数据，返回聚类中心")
            # 返回该功能群对应的聚类中心
            if group in self.group_to_cluster:
                cluster_id = self.group_to_cluster[group]
                return self.cluster_centers[cluster_id]
            else:
                raise ValueError(f"功能群'{group}'没有对应的聚类中心")
        
        # 获取对应功能群的聚类中心特征
        if group in self.group_to_cluster:
            cluster_id = self.group_to_cluster[group]
            return self.cluster_centers[cluster_id]
        else:
            # 回退：找到第一个属于该功能群的真菌的聚类标签
            for i, g in enumerate(self.functional_groups):
                if g == group:
                    cluster_id = self.cluster_labels[i]
                    return self.cluster_centers[cluster_id]
            
            # 如果还是没有找到，返回空数组
            return np.array([])
    
    def get_fungus_group_map(self) -> dict:
        """
        获取真菌ID到功能群的映射字典
        
        :param self: 类实例本身
        :return: 真菌ID到功能群的映射字典
        :rtype: dict
        """
        if self.functional_groups is None:
            raise ValueError("请先调用fit方法进行聚类拟合")
        
        # 创建一个字典，键是真菌ID，值是功能群标签
        return dict(zip(self.fungus_ids, self.functional_groups))
    
    def get_cluster_statistics(self) -> dict:
        """
        获取各聚类的统计信息
        
        :param self: 类实例本身
        :return: 聚类统计信息
        :rtype: dict
        """
        if self.cluster_labels is None:
            raise ValueError("请先调用fit方法进行聚类拟合")
        
        stats = {}
        for cluster_id in range(self.n_clusters):
            # 找到属于该聚类的真菌索引
            indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # 统计功能群分布
            group_counts = {}
            for i in indices:
                group = self.functional_groups[i]
                group_counts[group] = group_counts.get(group, 0) + 1
            
            stats[f"聚类{cluster_id}"] = {
                "数量": len(indices),
                "功能群分布": group_counts
            }
        
        return stats
    
# 单元测试
if __name__ == "__main__":
    import pandas as pd
    from data.data_loader import FungalDataLoader

    # 生成更符合实际的模拟特征空间
    # 创建三类明显不同的真菌特征
    np.random.seed(42)  # 确保可重复性
    
    n_fungi = 34
    mock_data = {
        "fungus_id": [f"F{i}" for i in range(1, n_fungi + 1)],
    }
    
    # 创建三类明显不同的特征
    # F型：高分解率，高菌丝延伸率
    # I型：中等特征
    # S型：低分解率，低菌丝延伸率
    
    # 分配每个真菌的类型（12个F，10个I，12个S）
    fungus_types = ['F'] * 12 + ['I'] * 10 + ['S'] * 12
    
    # 基于类型生成特征
    p_10, p_16, p_22 = [], [], []
    mu_10, mu_16, mu_22 = [], [], []
    Q10 = []
    enzyme_1, enzyme_2, enzyme_3, enzyme_4, enzyme_5 = [], [], [], [], []
    
    for f_type in fungus_types:
        if f_type == 'F':  # 快速分解型
            p_10.append(np.random.uniform(40, 50))
            p_16.append(np.random.uniform(45, 55))
            p_22.append(np.random.uniform(50, 60))
            mu_10.append(np.random.uniform(4, 5))
            mu_16.append(np.random.uniform(5, 6))
            mu_22.append(np.random.uniform(6, 7))
            Q10.append(np.random.uniform(1.5, 2.0))
            enzyme_1.append(np.random.uniform(0.7, 1.0))
            enzyme_2.append(np.random.uniform(0.6, 0.9))
            enzyme_3.append(np.random.uniform(0.8, 1.0))
            enzyme_4.append(np.random.uniform(0.7, 0.9))
            enzyme_5.append(np.random.uniform(0.6, 0.8))
        elif f_type == 'S':  # 慢速分解型
            p_10.append(np.random.uniform(10, 20))
            p_16.append(np.random.uniform(15, 25))
            p_22.append(np.random.uniform(20, 30))
            mu_10.append(np.random.uniform(1, 2))
            mu_16.append(np.random.uniform(1.5, 2.5))
            mu_22.append(np.random.uniform(2, 3))
            Q10.append(np.random.uniform(2.5, 3.0))
            enzyme_1.append(np.random.uniform(0.1, 0.4))
            enzyme_2.append(np.random.uniform(0.2, 0.5))
            enzyme_3.append(np.random.uniform(0.1, 0.3))
            enzyme_4.append(np.random.uniform(0.2, 0.4))
            enzyme_5.append(np.random.uniform(0.3, 0.5))
        else:  # 中间型
            p_10.append(np.random.uniform(25, 35))
            p_16.append(np.random.uniform(30, 40))
            p_22.append(np.random.uniform(35, 45))
            mu_10.append(np.random.uniform(2.5, 3.5))
            mu_16.append(np.random.uniform(3.0, 4.0))
            mu_22.append(np.random.uniform(3.5, 4.5))
            Q10.append(np.random.uniform(2.0, 2.5))
            enzyme_1.append(np.random.uniform(0.4, 0.7))
            enzyme_2.append(np.random.uniform(0.4, 0.7))
            enzyme_3.append(np.random.uniform(0.3, 0.6))
            enzyme_4.append(np.random.uniform(0.4, 0.6))
            enzyme_5.append(np.random.uniform(0.4, 0.7))
    
    # 添加一些随机扰动
    mock_data["p_10"] = np.array(p_10) + np.random.normal(0, 2, n_fungi)
    mock_data["p_16"] = np.array(p_16) + np.random.normal(0, 2, n_fungi)
    mock_data["p_22"] = np.array(p_22) + np.random.normal(0, 2, n_fungi)
    mock_data["mu_10"] = np.array(mu_10) + np.random.normal(0, 0.3, n_fungi)
    mock_data["mu_16"] = np.array(mu_16) + np.random.normal(0, 0.3, n_fungi)
    mock_data["mu_22"] = np.array(mu_22) + np.random.normal(0, 0.3, n_fungi)
    mock_data["Q10"] = np.array(Q10) + np.random.normal(0, 0.1, n_fungi)
    mock_data["enzyme_1"] = np.array(enzyme_1) + np.random.normal(0, 0.05, n_fungi)
    mock_data["enzyme_2"] = np.array(enzyme_2) + np.random.normal(0, 0.05, n_fungi)
    mock_data["enzyme_3"] = np.array(enzyme_3) + np.random.normal(0, 0.05, n_fungi)
    mock_data["enzyme_4"] = np.array(enzyme_4) + np.random.normal(0, 0.05, n_fungi)
    mock_data["enzyme_5"] = np.array(enzyme_5) + np.random.normal(0, 0.05, n_fungi)
    
    # 确保所有值在合理范围内
    for key in ["enzyme_1", "enzyme_2", "enzyme_3", "enzyme_4", "enzyme_5"]:
        mock_data[key] = np.clip(mock_data[key], 0.0, 1.0)
    
    # 创建DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    print("模拟数据特征统计:")
    print(mock_df.describe())
    
    # 使用真实的数据加载器
    loader = FungalDataLoader()
    loader.raw_data = mock_df
    loader.clean_data()
    feature_space = loader.build_feature_space()
    fungus_ids = loader.fungus_ids

    # 测试功能群分类
    print("\n=== 功能群分类测试 ===")
    classifier = FunctionalGroupClassifier(n_clusters=3)
    classifier.fit(feature_space, fungus_ids)
    
    # 输出结果
    print("\n=== 功能群分类测试结果 ===")
    group_map = classifier.get_fungus_group_map()
    print("真菌功能群分布（前10个）:")
    for i, (fungus_id, group) in enumerate(list(group_map.items())[:10]):
        print(f"  {fungus_id}: {group}")
    
    # 检查各功能群的特征中心
    for group in ['F', 'S', 'I']:
        try:
            traits = classifier.get_group_traits(group)
            print(f"\n{group}型特征中心（标准化后）:")
            print(f"  p_10: {traits[0]:.3f}, p_16: {traits[1]:.3f}, p_22: {traits[2]:.3f}")
            print(f"  mu_10: {traits[3]:.3f}, mu_16: {traits[4]:.3f}, mu_22: {traits[5]:.3f}")
            print(f"  Q10: {traits[6]:.3f}")
        except Exception as e:
            print(f"获取{group}型特征中心时出错: {e}")
    
    # 显示聚类统计
    print("\n=== 聚类统计 ===")
    stats = classifier.get_cluster_statistics()
    for cluster, info in stats.items():
        print(f"{cluster}: {info['数量']}个真菌, 功能群分布: {info['功能群分布']}")