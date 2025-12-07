# calibration/parameter_optim.py
# 基于 Adam 优化算法，校准fungus/traits中的核心参数

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
from fungus.traits import FungalTraitManager
from model.decomposition import DecompositionDynamics
from model.growth import FungalGrowthModel
from data.environment_db import EnvironmentGenerator

class AdamParameterOptimizer:
    """
    Adam参数优化器（上层核心）
    
    实现理论模型3.1节：基于实验数据校准真菌性状参数（α、μ、Q10等）
    
    依赖：中层模型（分解/生长）、底层环境生成器、Adam优化算法
    """
    def __init__(
        self,
        trait_manager: FungalTraitManager,
        decomp_model: DecompositionDynamics,
        growth_model: FungalGrowthModel,
        env_gen: EnvironmentGenerator,
        lr: float = 0.001,  # 学习率
        beta1: float = 0.9, # Adam一阶矩系数
        beta2: float = 0.999,# Adam二阶矩系数
        eps: float = 1e-8   # 防止除零
    ):
        self.trait_manager = trait_manager
        self.decomp_model = decomp_model
        self.growth_model = growth_model
        self.env_gen = env_gen

        # Adam优化超参数
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # 待优化参数（扁平化存储，便于梯度计算）
        self.params = self._flatten_params()  # [F_alpha, F_mu, F_Q10_alpha, ..., S_alpha, ...]
        self.m = np.zeros_like(self.params)   # 一阶矩
        self.v = np.zeros_like(self.params)   # 二阶矩
        self.t = 0  # 迭代步数

    def _flatten_params(self) -> np.ndarray:
        """将性状参数扁平化为一维数组（便于优化）"""
        params_list = []
        # 遍历F/S/I三类的核心参数
        for group in ["F", "S", "I"]:
            params_list.append(self.trait_manager.traits[group]["alpha_ref"])
            params_list.append(self.trait_manager.traits[group]["mu_ref"])
            params_list.append(self.trait_manager.traits[group]["Q10_alpha"])
            params_list.append(self.trait_manager.traits[group]["Q10_mu"])
        return np.array(params_list)

    def _unflatten_params(self, flat_params: np.ndarray) -> None:
        """将扁平化参数还原到trait_manager"""
        idx = 0
        for group in ["F", "S", "I"]:
            self.trait_manager.update_traits(group, {"alpha_ref": flat_params[idx]})
            idx += 1
            self.trait_manager.update_traits(group, {"mu_ref": flat_params[idx]})
            idx += 1
            self.trait_manager.update_traits(group, {"Q10_alpha": flat_params[idx]})
            idx += 1
            self.trait_manager.update_traits(group, {"Q10_mu": flat_params[idx]})
            idx += 1

    def _simulate_with_params(self, flat_params: np.ndarray, F_init: dict, W_init: float) -> tuple[np.ndarray, dict]:
        """用给定参数运行一次仿真，返回W_history和F_history"""
        # 临时更新参数
        self._unflatten_params(flat_params)
        
        # 重置模型历史
        self.decomp_model.W_history = []
        self.growth_model.F_history = {}
        
        # 运行仿真
        W = W_init
        F = F_init.copy()
        M, T = self.env_gen.M, self.env_gen.T
        for day in range(len(M)):
            F = self.growth_model.step(F, T[day], M[day])
            W = self.decomp_model.step(W, F, T[day])
        
        # 返回结果
        W_history = np.array(self.decomp_model.W_history)
        F_history = {fid: np.array(vals) for fid, vals in self.growth_model.F_history.items()}
        return W_history, F_history

    def _loss_function(self, flat_params: np.ndarray, F_init: dict, W_init: float, W_obs: np.ndarray) -> float:
        """损失函数：模型预测W vs 实验观测W的MSE"""
        W_pred, _ = self._simulate_with_params(flat_params, F_init, W_init)
        # 对齐长度（取较短的）
        min_len = min(len(W_pred), len(W_obs))
        loss = np.mean((W_pred[:min_len] - W_obs[:min_len])**2)
        return loss

    def _compute_gradient(self, F_init: dict, W_init: float, W_obs: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """数值梯度计算（中心差分）"""
        grad = np.zeros_like(self.params)
        for i in range(len(self.params)):
            # 正向扰动
            params_plus = self.params.copy()
            params_plus[i] += h
            loss_plus = self._loss_function(params_plus, F_init, W_init, W_obs)
            
            # 反向扰动
            params_minus = self.params.copy()
            params_minus[i] -= h
            loss_minus = self._loss_function(params_minus, F_init, W_init, W_obs)
            
            # 中心差分
            grad[i] = (loss_plus - loss_minus) / (2 * h)
        return grad

    def step(self, F_init: dict, W_init: float, W_obs: np.ndarray) -> tuple[float, np.ndarray]:
        """执行一步Adam优化"""
        self.t += 1
        
        # 计算梯度
        grad = self._compute_gradient(F_init, W_init, W_obs)
        
        # Adam更新
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差校正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 更新参数
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # 还原参数到trait_manager
        self._unflatten_params(self.params)
        
        # 计算当前损失
        loss = self._loss_function(self.params, F_init, W_init, W_obs)
        return loss, self.params

    def optimize(self, F_init: dict, W_init: float, W_obs: np.ndarray, max_iter: int = 100) -> dict:
        """批量优化参数"""
        loss_history = []
        params_history = []
        
        print("=== 开始参数校准 ===")
        for iter in range(max_iter):
            loss, params = self.step(F_init, W_init, W_obs)
            loss_history.append(loss)
            params_history.append(params.copy())
            
            if (iter + 1) % 10 == 0:
                print(f"迭代{iter+1}/{max_iter} | 损失：{loss:.4f}")
        
        # 返回优化结果
        return {
            "loss_history": loss_history,
            "params_history": params_history,
            "final_params": self.params,
            "final_loss": loss
        }

# 单元测试
if __name__ == "__main__":
    import pandas as pd
    from data import FungalDataLoader, EnvironmentGenerator
    from fungus import FunctionalGroupClassifier, FungalTraitManager
    from model import DecompositionDynamics, FungalGrowthModel

    # 初始化依赖
    # 真菌数据
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

    # 环境+模型
    env_gen = EnvironmentGenerator(duration_days=20)
    env_gen.generate_environment_series()
    decomp_model = DecompositionDynamics(trait_manager, initial_W=100.0)
    growth_model = FungalGrowthModel(trait_manager, K_F=10.0)

    # 模拟实验观测数据（带噪声）
    F_init = {"F1": 2.0, "F2": 1.0, "F3": 1.5}
    W_init = 100.0
    # 先运行一次得到"真实"观测（加噪声）
    W_true, _ = AdamParameterOptimizer(trait_manager, decomp_model, growth_model, env_gen)._simulate_with_params(
        AdamParameterOptimizer(trait_manager, decomp_model, growth_model, env_gen).params, F_init, W_init
    )
    W_obs = W_true + np.random.normal(0, 2.0, len(W_true))  # 加噪声

    # 初始化优化器
    optimizer = AdamParameterOptimizer(trait_manager, decomp_model, growth_model, env_gen, lr=0.01)

    # 运行优化
    result = optimizer.optimize(F_init, W_init, W_obs, max_iter=50)

    # 输出结果
    print("\n=== 参数校准测试结果 ===")
    print(f"初始损失：{result['loss_history'][0]:.4f} | 最终损失：{result['final_loss']:.4f}")
    print(f"优化后F型α_ref：{trait_manager.traits['F']['alpha_ref']:.4f}（初始0.05）")
    print(f"优化后S型μ_ref：{trait_manager.traits['S']['mu_ref']:.4f}（初始0.02）")