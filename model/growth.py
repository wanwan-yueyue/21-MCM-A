# model/growth.py
# 真菌生长模型，实现真菌生物量增长方程，考虑环境（温度 / 湿度）和资源限制

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

class FungalGrowthModel:
    """
    真菌生长模型

    实现理论模型：dF_i/dt = μ_i(T,M) * F_i * (1 - F_i/K_F - Σ(φ_ij * F_j/K_F))

    其中：μ_i(T,M)为温度/湿度校正的生长速率，φ_ij为净交互系数，K_F为承载量
    """
    def __init__(
        self,
        trait_manager: FungalTraitManager,
        K_F: float = 10.0,  # 真菌承载量（可校准）
        humidity_sensitivity: float = 0.5  # 湿度对生长的影响系数
    ):
        self.trait_manager = trait_manager
        self.K_F = K_F
        self.humidity_sensitivity = humidity_sensitivity
        self.F_history = {}  # 各真菌生物量历史 {fungus_id: [历史值]}

    def _mu_correction(self, mu_T: float, M: float) -> float:
        """湿度校正生长速率：μ(T,M) = μ(T) * (1 - exp(-s*M))"""
        return mu_T * (1 - np.exp(-self.humidity_sensitivity * M))

    def dF_dt(self, fungus_id: str, F_i: float, F_biomass: dict, T: float, M: float) -> float:
        """
        计算单个真菌的生物量变化率

        :param fungus_id: 目标真菌ID
        :param F_i: 目标真菌当前生物量
        :param F_biomass: 所有真菌生物量字典
        :param T: 环境温度
        :param M: 环境湿度
        :return: dF_i/dt
        """
        if F_i <= 0:
            return 0.0  # 无生物量时停止生长
        
        # 温度校正生长速率μ_i(T)
        mu_T = self.trait_manager.get_fungus_trait(fungus_id, "mu_ref", T)
        # 湿度校正μ_i(T,M)
        mu = self._mu_correction(mu_T, M)
        if mu <= 0:
            return 0.0
        
        # 自身限制项：1 - F_i/K_F
        self_limit = 1 - F_i / self.K_F
        if self_limit <= 0:
            return 0.0
        
        # 种间交互项：Σ(φ_ij * F_j/K_F)
        interaction_sum = 0.0
        group_i = self.trait_manager.group_classifier.get_fungus_group_map()[fungus_id]
        for fungus_j, F_j in F_biomass.items():
            if fungus_j == fungus_id:
                continue
            group_j = self.trait_manager.group_classifier.get_fungus_group_map()[fungus_j]
            beta, gamma = self.trait_manager.get_interaction_coeffs(group_i, group_j)
            phi_ij = gamma - beta  # 净交互系数
            interaction_sum += phi_ij * F_j / self.K_F
        
        # 总生长速率
        dF = mu * F_i * (self_limit - interaction_sum)
        return dF

    def step(self, F_biomass: dict, T: float, M: float, dt: float = 1.0) -> dict:
        """
        单步更新所有真菌的生物量
        
        :param F_biomass: 初始生物量字典
        :param T: 环境温度
        :param M: 环境湿度
        :param dt: 时间步长
        :return: 新的生物量字典
        """
        new_F = {}
        for fungus_id, F_i in F_biomass.items():
            dF = self.dF_dt(fungus_id, F_i, F_biomass, T, M)
            F_new = F_i + dF * dt
            F_new = max(F_new, 0.0)  # 生物量不能为负
            new_F[fungus_id] = F_new
            
            # 记录历史
            if fungus_id not in self.F_history:
                self.F_history[fungus_id] = []
            self.F_history[fungus_id].append(F_new)
        
        return new_F

# 单元测试
if __name__ == "__main__":
    import pandas as pd
    from data.data_loader import FungalDataLoader
    from data.environment_db import EnvironmentGenerator
    from fungus.functional_group import FunctionalGroupClassifier
    from fungus.traits import FungalTraitManager

    # 初始化依赖
    mock_data = pd.DataFrame({
        "fungus_id": [f"F{i}" for i in range(1, 4)],
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

    # 测试生长模型
    growth_model = FungalGrowthModel(trait_manager, K_F=10.0)
    F_biomass = {"F1": 2.0, "F2": 1.0, "F3": 1.5}  # 初始生物量

    print("\n=== 真菌生长模型测试结果 ===")
    current_F = F_biomass
    for day in range(10):
        T = env_gen.T[day]
        M = env_gen.M[day]
        current_F = growth_model.step(current_F, T, M)
        print(f"第{day+1}天（T={T:.1f}℃, M={M:.2f}）：")
        for fid, f in current_F.items():
            print(f"  {fid}生物量：{f:.2f}")