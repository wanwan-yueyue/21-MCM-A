# fungus/traits.py
# 真菌性状参数管理

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
from utils.metrics import q10_correction
from fungus.functional_group import FunctionalGroupClassifier

class FungalTraitManager:
    """
    真菌性状参数管理器（中层核心）
    功能：存储/校准/查询不同功能群的核心性状参数，支持温度/湿度校正
    依赖：functional_group分类结果、utils的Q10校正
    """
    # 功能群默认性状参数（基于理论模型初始值）
    DEFAULT_TRAITS = {
        "F": {  # 快速型
            "alpha_ref": 0.05,   # 22℃分解系数α
            "mu_ref": 0.1,       # 22℃生长速率μ
            "Q10_alpha": 2.5,    # 分解系数Q10
            "Q10_mu": 2.0,       # 生长速率Q10
            "enzyme_activity": np.array([0.8, 0.7, 0.9, 0.8, 0.7]),  # 5种酶活性
            "competition_coeff": 0.8,  # 竞争系数β
            "synergy_coeff": 0.2       # 协同系数γ
        },
        "S": {  # 慢速型
            "alpha_ref": 0.01,
            "mu_ref": 0.02,
            "Q10_alpha": 1.8,
            "Q10_mu": 1.5,
            "enzyme_activity": np.array([0.2, 0.3, 0.1, 0.2, 0.3]),
            "competition_coeff": 0.3,
            "synergy_coeff": 0.5
        },
        "I": {  # 中间型
            "alpha_ref": 0.03,
            "mu_ref": 0.06,
            "Q10_alpha": 2.2,
            "Q10_mu": 1.8,
            "enzyme_activity": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            "competition_coeff": 0.5,
            "synergy_coeff": 0.3
        }
    }

    def __init__(self, group_classifier: FunctionalGroupClassifier = None):
        self.group_classifier = group_classifier  # 功能群分类器实例
        self.traits = self.DEFAULT_TRAITS.copy()  # 初始化性状参数
        self.calibrated = False                   # 参数是否已校准

    def update_traits(self, group: str, params: dict) -> None:
        """更新某功能群的性状参数（用于后续校准）"""
        if group not in self.traits:
            raise ValueError("group仅支持F/S/I")
        self.traits[group].update(params)
        print(f"✅ 更新{group}型性状参数：{params.keys()}")

    def get_trait(self, group: str, trait_name: str, T: float = 22.0) -> float | np.ndarray:
        """
        获取某功能群的性状参数（支持温度校正）
        :param group: 功能群（F/S/I）
        :param trait_name: 参数名（如alpha_ref/mu_ref/enzyme_activity）
        :param T: 环境温度（用于Q10校正）
        :return: 校正后的参数值
        """
        if group not in self.traits:
            raise ValueError("group仅支持F/S/I")
        if trait_name not in self.traits[group]:
            raise ValueError(f"{group}型无{trait_name}参数")
        
        # 基础参数值
        value = self.traits[group][trait_name]

        # 温度校正（仅对alpha和mu）
        if trait_name == "alpha_ref":
            return q10_correction(value, self.traits[group]["Q10_alpha"], t_current=T)
        elif trait_name == "mu_ref":
            return q10_correction(value, self.traits[group]["Q10_mu"], t_current=T)
        else:
            return value

    def get_fungus_trait(self, fungus_id: str, trait_name: str, T: float = 22.0) -> float | np.ndarray:
        """根据真菌ID获取其性状参数（衔接功能群分类）"""
        if self.group_classifier is None:
            raise ValueError("请先传入group_classifier实例")
        group_map = self.group_classifier.get_fungus_group_map()
        if fungus_id not in group_map:
            raise ValueError(f"未找到真菌ID：{fungus_id}")
        
        group = group_map[fungus_id]
        return self.get_trait(group, trait_name, T)

    def get_interaction_coeffs(self, group_i: str, group_j: str) -> tuple[float, float]:
        """获取两个功能群的竞争-协同系数（φ_ij = γ - β）"""
        beta = self.traits[group_i]["competition_coeff"] * self.traits[group_j]["competition_coeff"]
        gamma = self.traits[group_i]["synergy_coeff"] * self.traits[group_j]["synergy_coeff"]
        return beta, gamma

# 单元测试
if __name__ == "__main__":
    import pandas as pd
    from data.data_loader import FungalDataLoader
    from fungus.functional_group import FunctionalGroupClassifier

    # 初始化功能群分类器
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
    loader = FungalDataLoader()
    loader.raw_data = mock_data
    loader.clean_data()
    feature_space = loader.build_feature_space()
    classifier = FunctionalGroupClassifier()
    classifier.fit(feature_space, loader.fungus_ids)

    # 测试性状管理器
    trait_manager = FungalTraitManager(classifier)
    
    # 查询参数（温度校正）
    T = 32.0  # 32℃
    F_alpha = trait_manager.get_trait("F", "alpha_ref", T)
    S_mu = trait_manager.get_trait("S", "mu_ref", T)
    print("\n=== 性状参数测试结果 ===")
    print(f"32℃下F型分解系数α：{F_alpha:.4f}（22℃参考值0.05）")
    print(f"32℃下S型生长速率μ：{S_mu:.4f}（22℃参考值0.02）")
    
    # 根据真菌ID查询
    fungus_id = "F1"
    enzyme = trait_manager.get_fungus_trait(fungus_id, "enzyme_activity")
    print(f"真菌{fungus_id}的酶活性：{enzyme}")
    
    # 交互系数
    beta, gamma = trait_manager.get_interaction_coeffs("F", "S")
    print(f"F-S型竞争系数β：{beta:.4f}，协同系数γ：{gamma:.4f}，净交互φ：{gamma-beta:.4f}")