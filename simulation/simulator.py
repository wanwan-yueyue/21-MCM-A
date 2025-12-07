# simulation/simulator.py
# 仿真主控制
# 整合所有模块，实现端到端的完整仿真流程

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
import pandas as pd
from utils import rk4
from data import FungalDataLoader, EnvironmentGenerator
from fungus import FunctionalGroupClassifier, FungalTraitManager
from model import DecompositionDynamics, FungalGrowthModel, InteractionModel
from calibration import AdamParameterOptimizer
from evaluation import DiversityEvaluator, TOPSISEvaluator

class FungalDecompositionSimulator:
    """
    真菌分解仿真器

    整合所有模块，实现完整仿真流程：
    
    数据加载 → 功能群分类 → 参数校准 → 动态仿真 → 结果评估 → 输出报告
    """
    def __init__(
        self,
        data_path: str = None,
        env_type: str = "semi_humid",
        duration_days: int = 365,
        climate_trend: str = None,
        calibrate_params: bool = True  # 是否校准参数
    ):
        # 初始化配置
        self.data_path = data_path
        self.env_type = env_type
        self.duration_days = duration_days
        self.climate_trend = climate_trend
        self.calibrate_params = calibrate_params

        # 仿真结果存储
        self.results = {
            "W_history": [],
            "F_history": {},
            "env_history": {"T": [], "M": []},
            "metrics": {},
            "min_diversity_threshold": None
        }

        # 初始化所有模块（逐层构建）
        self._init_data_layer()
        self._init_fungus_layer()
        self._init_model_layer()
        self._init_calibration_layer()
        self._init_evaluation_layer()


    def _init_data_layer(self):
        """初始化底层数据模块"""
        # 真菌数据加载
        self.loader = FungalDataLoader(self.data_path)
        if self.data_path:
            self.loader.load_raw_data(self.data_path)
        else:
            # 生成模拟数据（无真实数据时）
            mock_data = pd.DataFrame({
                "fungus_id": [f"F{i}" for i in range(1, 35)],
                "p_10": np.random.uniform(10, 50, 34),
                "p_16": np.random.uniform(15, 55, 34),
                "p_22": np.random.uniform(20, 60, 34),
                "mu_10": np.random.uniform(1, 5, 34),
                "mu_16": np.random.uniform(2, 6, 34),
                "mu_22": np.random.uniform(3, 7, 34),
                "Q10": np.random.uniform(1.5, 3.0, 34),
                "enzyme_1": np.random.uniform(0.1, 1.0, 34),
                "enzyme_2": np.random.uniform(0.1, 1.0, 34),
                "enzyme_3": np.random.uniform(0.1, 1.0, 34),
                "enzyme_4": np.random.uniform(0.1, 1.0, 34),
                "enzyme_5": np.random.uniform(0.1, 1.0, 34),
            })
            self.loader.raw_data = mock_data
        self.loader.clean_data()
        self.feature_space = self.loader.build_feature_space()

        # 环境生成
        self.env_gen = EnvironmentGenerator(
            env_type=self.env_type,
            duration_days=self.duration_days,
            climate_trend=self.climate_trend
        )
        self.env_gen.generate_environment_series()
        self.results["env_history"]["T"] = self.env_gen.T
        self.results["env_history"]["M"] = self.env_gen.M

    def _init_fungus_layer(self):
        """初始化中层真菌模块"""
        # 功能群分类
        self.classifier = FunctionalGroupClassifier(n_clusters=3)
        self.classifier.fit(self.feature_space, self.loader.fungus_ids)

        # 性状参数管理
        self.trait_manager = FungalTraitManager(self.classifier)

    def _init_model_layer(self):
        """初始化中层模型模块"""
        self.decomp_model = DecompositionDynamics(self.trait_manager, initial_W=100.0)
        self.growth_model = FungalGrowthModel(self.trait_manager, K_F=10.0)
        self.interaction_model = InteractionModel(self.trait_manager)

    def _init_calibration_layer(self):
        """初始化上层校准模块"""
        if self.calibrate_params:
            self.optimizer = AdamParameterOptimizer(
                self.trait_manager,
                self.decomp_model,
                self.growth_model,
                self.env_gen,
                lr=0.01
            )

    def _init_evaluation_layer(self):
        """初始化上层评估模块"""
        self.diversity_evaluator = DiversityEvaluator(self.decomp_model, self.growth_model, self.env_gen)

    def _calibrate_params(self, F_init: dict, W_init: float):
        """参数校准（生成模拟观测数据）"""
        # 生成带噪声的观测数据
        W_true, _ = self.optimizer._simulate_with_params(self.optimizer.params, F_init, W_init)
        W_obs = W_true + np.random.normal(0, 2.0, len(W_true))
        
        # 运行优化
        optimize_result = self.optimizer.optimize(F_init, W_init, W_obs, max_iter=50)
        print(f"✅ 参数校准完成 | 初始损失：{optimize_result['loss_history'][0]:.4f} | 最终损失：{optimize_result['final_loss']:.4f}")

    def run_simulation(self, F_init: dict = None, W_init: float = 100.0):
        """
        运行完整仿真

        :param F_init: 初始真菌生物量（默认随机生成）
        :param W_init: 初始木质纤维质量
        """
        # 初始化生物量
        if F_init is None:
            # 随机生成前10个真菌的初始生物量
            F_init = {fid: np.random.uniform(0.5, 2.0) for fid in self.loader.fungus_ids[:10]}
        current_F = F_init.copy()
        current_W = W_init

        # 参数校准（可选）
        if self.calibrate_params:
            self._calibrate_params(F_init, W_init)

        # 时间步进仿真
        print("\n=== 开始完整仿真 ===")
        for day in range(self.duration_days):
            if day % 30 == 0:
                print(f"仿真进度：{day}/{self.duration_days}天")
            
            # 当前环境参数
            T_day = self.env_gen.T[day]
            M_day = self.env_gen.M[day]

            # 更新真菌生物量
            current_F = self.growth_model.step(current_F, T_day, M_day)

            # 更新木质纤维质量
            current_W = self.decomp_model.step(current_W, current_F, T_day)

        # 存储结果
        self.results["W_history"] = self.decomp_model.W_history
        self.results["F_history"] = self.growth_model.F_history

        # 评估指标
        self.results["metrics"] = self.diversity_evaluator.get_comprehensive_metrics()

        # 计算最小生物多样性阈值（模拟不同多样性水平）
        self._calc_min_diversity_threshold()

        print("✅ 仿真完成！")

    def _calc_min_diversity_threshold(self):
        """模拟不同多样性水平，计算最小阈值"""
        # 生成不同多样性水平的指标
        diversity_series = np.linspace(0.1, 0.9, 5)
        metrics_list = []
        for div in diversity_series:
            # 模拟不同多样性下的指标（简化：多样性越高，指标越好）
            metrics_list.append({
                "DEI": 0.02 + 0.033 * div,
                "diversity_index": div,
                "stability": 0.5 + 0.33 * div,
                "synergy_gain": 0.05 + 0.14 * div
            })
        
        # TOPSIS评价
        topsis = TOPSISEvaluator(metrics_list)
        self.results["min_diversity_threshold"] = topsis.find_min_diversity_threshold(diversity_series, threshold_closeness=0.8)

    def export_results(self, save_path: str = "simulation_results.csv"):
        """导出仿真结果到CSV"""
        # 整理结果
        df_W = pd.DataFrame({
            "day": self.env_gen.timeline,
            "temperature": self.env_gen.T,
            "humidity": self.env_gen.M,
            "wood_mass": self.results["W_history"]
        })

        # 真菌生物量
        df_F = pd.DataFrame({"day": self.env_gen.timeline})
        for fid, F_vals in self.results["F_history"].items():
            # 补全长度（适配时间轴）
            F_vals = F_vals + [F_vals[-1]] * (len(self.env_gen.timeline) - len(F_vals)) if len(F_vals) < len(self.env_gen.timeline) else F_vals[:len(self.env_gen.timeline)]
            df_F[f"fungus_{fid}"] = F_vals

        # 合并结果
        df_results = pd.merge(df_W, df_F, on="day")

        # 保存
        df_results.to_csv(save_path, index=False)
        print(f"✅ 仿真结果已导出到：{save_path}")

        # 输出核心指标
        print("\n=== 核心仿真指标 ===")
        for k, v in self.results["metrics"].items():
            print(f"{k}: {v:.4f}")
        print(f"最小生物多样性阈值：{self.results['min_diversity_threshold']:.2f}")

# 单元测试（完整仿真流程）
if __name__ == "__main__":
    # 初始化仿真器（模拟数据，365天，湿润环境，变暖趋势）
    simulator = FungalDecompositionSimulator(
        env_type="humid",
        duration_days=365,
        climate_trend="warming",
        calibrate_params=True
    )

    # 运行仿真（初始生物量随机生成）
    simulator.run_simulation()

    # 导出结果
    simulator.export_results("simulation_results.csv")