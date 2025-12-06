# model/interaction.py
# 竞争 - 协同相互作用模型，细化种间交互逻辑，为生长模型提供交互系数的环境依赖性校正

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

class InteractionModel:
    """
    竞争-协同相互作用模型

    实现理论模型5.1-5.3节：φ_ij(M,T) = γ_ij - β_ij(M,T,R_i/R_j)
    
    其中：β_ij依赖环境（M/T）和资源利用效率比（R_i/R_j）
    """
    def __init__(
        self,
        trait_manager: FungalTraitManager,
        resource_sensitivity: float = 0.3  # 资源效率对竞争的影响系数
    ):
        self.trait_manager = trait_manager
        self.resource_sensitivity = resource_sensitivity

    def _resource_efficiency(self, group: str, T: float, M: float) -> float:
        """计算功能群的资源利用效率R：R = α_i(T) * enzyme_activity_mean * M"""
        alpha = self.trait_manager.get_trait(group, "alpha_ref", T)
        enzyme = self.trait_manager.get_trait(group, "enzyme_activity")
        enzyme_mean = np.mean(enzyme)
        R = alpha * enzyme_mean * M
        return max(R, 1e-8)  # 避免除零

    def _beta_correction(self, beta_ij: float, T: float, M: float, group_i: str, group_j: str) -> float:
        """
        环境依赖的竞争系数校正：β_ij(M,T) = β_ij * f(T,M) * (R_j/R_i)^s

        :param beta_ij: 基础竞争系数
        :param T: 环境温度
        :param M: 环境湿度
        :param group_i: 真菌i的功能群
        :param group_j: 真菌j的功能群
        :return: 校正后的β_ij
        """
        # 温度-湿度联合校正项f(T,M)
        T_opt = 22.0  # 最优温度
        M_opt = 0.6   # 最优湿度
        f_T = np.exp(-0.5 * ((T - T_opt)/5)**2)  # 温度高斯校正
        f_M = np.exp(-0.5 * ((M - M_opt)/0.2)**2) # 湿度高斯校正
        f_TM = f_T * f_M

        # 资源效率比
        R_i = self._resource_efficiency(group_i, T, M)
        R_j = self._resource_efficiency(group_j, T, M)
        R_ratio = (R_j / R_i) ** self.resource_sensitivity

        # 最终竞争系数
        beta_corrected = beta_ij * f_TM * R_ratio
        return beta_corrected

    def get_net_interaction(self, fungus_i: str, fungus_j: str, T: float, M: float) -> float:
        """
        计算两个真菌的净交互系数φ_ij = γ_ij - β_ij(M,T)

        :param fungus_i: 真菌i的ID
        :param fungus_j: 真菌j的ID
        :param T: 环境温度
        :param M: 环境湿度
        :return: 净交互系数（正=协同，负=竞争）
        """
        # 获取功能群
        group_map = self.trait_manager.group_classifier.get_fungus_group_map()
        group_i = group_map[fungus_i]
        group_j = group_map[fungus_j]

        # 基础交互系数
        beta_base, gamma_base = self.trait_manager.get_interaction_coeffs(group_i, group_j)

        # 校正竞争系数
        beta = self._beta_correction(beta_base, T, M, group_i, group_j)

        # 净交互系数
        phi_ij = gamma_base - beta
        return phi_ij

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

    # 测试交互模型
    interaction_model = InteractionModel(trait_manager)
    env_gen = EnvironmentGenerator(duration_days=1)
    env_gen.generate_environment_series()
    T, M = 30.0, 0.7  # 高温高湿

    # 计算净交互系数
    phi_12 = interaction_model.get_net_interaction("F1", "F2", T, M)
    phi_13 = interaction_model.get_net_interaction("F1", "F3", T, M)
    print("\n=== 交互模型测试结果 ===")
    print(f"F1-F2净交互系数φ：{phi_12:.4f}（正=协同，负=竞争）")
    print(f"F1-F3净交互系数φ：{phi_13:.4f}")

    # 不同环境下的交互变化
    T2, M2 = 15.0, 0.2  # 低温低湿
    phi_12_2 = interaction_model.get_net_interaction("F1", "F2", T2, M2)
    print(f"低温低湿下F1-F2净交互系数φ：{phi_12_2:.4f}（环境依赖性验证）")