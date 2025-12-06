# model/decomposition.py
# 实现理论模型的核心分解方程(分解动力学模型)

# 导入项目根目录到sys.path，确保可以导入data, fungus模块（快速调试用）
import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（21-A-final/）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path
sys.path.append(root_dir)

import numpy as np
from fungus.traits import FungalTraitManager
from data.environment_db import EnvironmentGenerator

class DecompositionDynamics:
    """
    木质纤维分解动力学模型

    实现理论模型4.1节：dW/dt = -Σ(α_i(T) * F_i * W * (W/(K_W + W)))

    依赖：fungus的性状参数、utils的Q10校正、data的环境参数
    """
    def __init__(
        self,
        trait_manager: FungalTraitManager,
        K_W: float = 50.0,  # 木质纤维半饱和常数
        initial_W: float = 100.0  # 初始木质纤维质量
    ):
        self.trait_manager = trait_manager
        self.K_W = K_W  # 半饱和常数（可校准）
        self.initial_W = initial_W
        self.W_history = []  # 木质纤维质量历史

    def dW_dt(self, W: float, F_biomass: dict, T: float) -> float:
        """
        计算木质纤维质量变化率

        :param W: 当前木质纤维质量
        :param F_biomass: 各真菌生物量字典 {fungus_id: 生物量}
        :param T: 当前环境温度
        :return: dW/dt
        """
        if W <= 0:
            return 0.0  # 无木质纤维时分解停止
        
        total_contribution = 0.0
        for fungus_id, F_i in F_biomass.items():
            # 获取该真菌的温度校正分解系数α_i(T)
            alpha_i = self.trait_manager.get_fungus_trait(fungus_id, "alpha_ref", T)
            # 分解动力学项（一级动力学+半饱和）
            term = alpha_i * F_i * W * (W / (self.K_W + W))
            total_contribution += term
        
        # 分解导致质量减少，故为负
        dW = -total_contribution
        return dW

    def step(self, W_prev: float, F_biomass: dict, T: float, dt: float = 1.0) -> float:
        """
        单步更新木质纤维质量（欧拉法，或调用utils的RK4）
        
        :param W_prev: 上一时刻质量
        :param F_biomass: 真菌生物量
        :param T: 环境温度
        :param dt: 时间步长（天）
        :return: 新的木质纤维质量
        """
        dW = self.dW_dt(W_prev, F_biomass, T)
        W_new = W_prev + dW * dt
        W_new = max(W_new, 0.0)  # 质量不能为负
        self.W_history.append(W_new)
        return W_new

# 单元测试
if __name__ == "__main__":
    import pandas as pd
    from data.data_loader import FungalDataLoader
    from data.environment_db import EnvironmentGenerator
    from fungus.functional_group import FunctionalGroupClassifier
    from fungus.traits import FungalTraitManager

    # 初始化依赖
    # 数据加载+功能群分类
    mock_data = pd.DataFrame({
        "fungus_id": [f"F{i}" for i in range(1, 4)],  # 简化为3种真菌
        "p_10": np.random.uniform(10, 50, 3),
        "p_16": np.random.uniform(15, 55, 3),
        "p_22": np.random.uniform(20, 60, 3),
        "mu_10": np.random.uniform(1, 5, 3),
        "mu_16": np.random.uniform(2, 6, 3),
        "mu_22": np.random.uniform(3, 7, 3),
        "Q10": np.random.uniform(1.5, 3.0, 3),
        "enzyme_1": np.random.uniform(0.1, 1.0, 3),
        "enzyme_2": np.random.uniform(0.1, 1.0, 3),
        "enzyme_3": np.random.uniform(0.1, 1.0, 3),
        "enzyme_4": np.random.uniform(0.1, 1.0, 3),
        "enzyme_5": np.random.uniform(0.1, 1.0, 3),
    })
    loader = FungalDataLoader()
    loader.raw_data = mock_data
    loader.clean_data()
    feature_space = loader.build_feature_space()
    classifier = FunctionalGroupClassifier()
    classifier.fit(feature_space, loader.fungus_ids)
    trait_manager = FungalTraitManager(classifier)

    # 环境生成器
    env_gen = EnvironmentGenerator(duration_days=10)
    env_gen.generate_environment_series()

    # 测试分解动力学
    decomp_model = DecompositionDynamics(trait_manager, initial_W=100.0)
    F_biomass = {"F1": 5.0, "F2": 3.0, "F3": 4.0}  # 模拟真菌生物量
    W_prev = decomp_model.initial_W

    print("\n=== 分解动力学测试结果 ===")
    for day in range(10):
        T = env_gen.T[day]
        W_new = decomp_model.step(W_prev, F_biomass, T)
        print(f"第{day+1}天（T={T:.1f}℃）：木质纤维质量 {W_new:.2f}（变化率 {W_new-W_prev:.2f}）")
        W_prev = W_new