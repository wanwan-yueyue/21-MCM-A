# fungal_decomposition/__init__.py
"""
木质纤维真菌分解仿真框架

核心功能：真菌功能群分类、分解动力学仿真、参数校准、多样性评估
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Ada"

# 统一暴露核心类（按使用频率排序）
from .simulation.simulator import FungalDecompositionSimulator
from .data.data_loader import FungalDataLoader
from .data.environment_db import EnvironmentGenerator
from .fungus.functional_group import FunctionalGroupClassifier
from .fungus.traits import FungalTraitManager
from .model.decomposition import DecompositionDynamics
from .model.growth import FungalGrowthModel
from .calibration.parameter_optim import AdamParameterOptimizer
from .evaluation.diversity_index import DiversityEvaluator
from .evaluation.topsis import TOPSISEvaluator

# 暴露核心工具函数
from .utils.numerical import rk4
from .utils.metrics import q10_correction, normalize_feature

# 定义对外公开的接口列表
__all__ = [
    # 核心仿真器（最常用）
    "FungalDecompositionSimulator",
    # 数据层
    "FungalDataLoader",
    "EnvironmentGenerator",
    # 真菌层
    "FunctionalGroupClassifier",
    "FungalTraitManager",
    # 模型层
    "DecompositionDynamics",
    "FungalGrowthModel",
    # 校准层
    "AdamParameterOptimizer",
    # 评估层
    "DiversityEvaluator",
    "TOPSISEvaluator",
    # 工具函数
    "rk4",
    "q10_correction",
    "normalize_feature"
]