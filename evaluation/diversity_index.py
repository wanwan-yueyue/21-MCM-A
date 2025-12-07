# evaluation/diversity_index.py
# 多样性 / 效率评估模块。
# 实现理论模型 7.1 节的核心评估指标（DEI、稳定性、多样性指数）

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
from model.decomposition import DecompositionDynamics
from model.growth import FungalGrowthModel
from data.environment_db import EnvironmentGenerator

class DiversityEvaluator:
    """
    分解效率与多样性评估器（上层核心）
    
    实现理论模型7.1节：DEI（分解效率）、I（多样性）、S（稳定性）、G（协同增益）
    
    依赖：中层模型的历史数据（W_history、F_history）
    """
    def __init__(
        self,
        decomp_model: DecompositionDynamics,
        growth_model: FungalGrowthModel,
        env_gen: EnvironmentGenerator
    ):
        self.decomp_model = decomp_model
        self.growth_model = growth_model
        self.env_gen = env_gen
        self.M = env_gen.M
        self.T = env_gen.T
        self.timeline = env_gen.timeline

    def calc_DEI(self, S_E: float = 1.0) -> float:
        """
        分解效率系数DEI = (1/T) ∫( (dW/dt)/W0 * S_E ) dt
        
        :param S_E: 环境适配系数（默认1.0）
        :return: DEI值（越大分解效率越高）
        """
        W_history = np.array(self.decomp_model.W_history)
        W0 = self.decomp_model.initial_W
        
        # 计算dW/dt（差分）
        dW_dt = np.diff(W_history) / (self.timeline[1] - self.timeline[0])
        dW_dt = np.concatenate([[0], dW_dt])  # 补全初始值
        
        # 积分计算
        integral = np.trapezoid((-dW_dt / W0) * S_E, self.timeline)
        DEI = integral / (self.timeline[-1] - self.timeline[0])
        return DEI

    def calc_diversity_index(self) -> float:
        """
        真菌多样性指数I = -Σ(p_i * ln(p_i))，p_i = F_i / ΣF_j
        :return: 多样性指数（越大多样性越高）
        """
        F_history = self.growth_model.F_history
        # 取最后时刻的生物量计算
        last_F = np.array([vals[-1] for vals in F_history.values()])
        total_F = np.sum(last_F)
        if total_F == 0:
            return 0.0
        
        p_i = last_F / total_F
        p_i = p_i[p_i > 0]  # 排除零值
        I = -np.sum(p_i * np.log(p_i))
        return I

    def calc_stability(self, DEI_window: int = 5, V_E: float = 1.0) -> float:
        """
        分解稳定性指数S = 1 - (σ_DEI / (μ_DEI * V_E))
        
        :param DEI_window: 滑动窗口大小
        :param V_E: 环境波动系数
        :return: 稳定性指数（越接近1越稳定）
        """
        W_history = np.array(self.decomp_model.W_history)
        W0 = self.decomp_model.initial_W
        dW_dt = np.diff(W_history) / (self.timeline[1] - self.timeline[0])
        dW_dt = np.concatenate([[0], dW_dt])
        
        # 计算滑动窗口DEI
        DEI_series = []
        for i in range(len(self.timeline) - DEI_window + 1):
            window_t = self.timeline[i:i+DEI_window]
            window_dW = dW_dt[i:i+DEI_window]
            integral = np.trapezoid((-window_dW / W0), window_t)
            DEI_window_val = integral / (window_t[-1] - window_t[0])
            DEI_series.append(DEI_window_val)
        
        if len(DEI_series) < 2:
            return 1.0
        
        # 计算稳定性
        sigma = np.std(DEI_series)
        mu = np.mean(DEI_series)
        if mu == 0:
            return 0.0
        
        S = 1 - (sigma / (mu * V_E))
        return max(S, 0.0)  # 稳定性≥0

    def calc_synergy_gain(self) -> float:
        """
        协同增益指数G = (DEI_total - ΣDEI_single) / ΣDEI_single
        
        （需额外运行单真菌仿真，此处简化为相对值）
        """
        DEI_total = self.calc_DEI()
        # 模拟单真菌DEI（简化：取各真菌生物量占比×总DEI）
        F_history = self.growth_model.F_history
        last_F = np.array([vals[-1] for vals in F_history.values()])
        total_F = np.sum(last_F)
        if total_F == 0:
            return 0.0
        
        DEI_single_sum = np.sum((last_F / total_F) * DEI_total * 0.8)  # 单真菌效率更低
        G = (DEI_total - DEI_single_sum) / DEI_single_sum if DEI_single_sum > 0 else 0.0
        return G

    def get_comprehensive_metrics(self) -> dict:
        """返回所有核心评估指标"""
        return {
            "DEI": self.calc_DEI(),
            "diversity_index": self.calc_diversity_index(),
            "stability": self.calc_stability(),
            "synergy_gain": self.calc_synergy_gain()
        }

# 单元测试
if __name__ == "__main__":
    import pandas as pd
    from data import FungalDataLoader, EnvironmentGenerator
    from fungus import FunctionalGroupClassifier, FungalTraitManager
    from model import DecompositionDynamics, FungalGrowthModel

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

    env_gen = EnvironmentGenerator(duration_days=20)
    env_gen.generate_environment_series()
    decomp_model = DecompositionDynamics(trait_manager, initial_W=100.0)
    growth_model = FungalGrowthModel(trait_manager, K_F=10.0)

    # 运行仿真
    F_init = {"F1": 2.0, "F2": 1.0, "F3": 1.5}
    W = 100.0
    for day in range(20):
        F_init = growth_model.step(F_init, env_gen.T[day], env_gen.M[day])
        W = decomp_model.step(W, F_init, env_gen.T[day])

    # 评估指标
    evaluator = DiversityEvaluator(decomp_model, growth_model, env_gen)
    metrics = evaluator.get_comprehensive_metrics()

    # 输出结果
    print("\n=== 多样性评估测试结果 ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")