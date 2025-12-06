# data/data_loader.py
# 真菌原始数据加载 / 预处理模块

# 导入项目根目录到sys.path，确保可以导入模块（快速调试用）
import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（21-A-final/）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path
sys.path.append(root_dir)

import pandas as pd
import numpy as np
from utils.metrics import fill_missing_data, normalize_feature

class FungalDataLoader:
    """
    真菌原始数据加载器（底层数据层）
    
    功能：加载，清洗，构建特征空间，为后续真菌功能群分类提供输入
    """
    def __init__(self, data_path:str = None):
        self.raw_data: pd.DataFrame = None      # 原始数据
        self.cleaned_data: pd.DataFrame = None      # 清洗后的数据
        self.feature_space: pd.DataFrame = None     # 标准化后的特征空间
        self.fungus_ids: np.ndarray = None      # 真菌 ID/名称 列表（对应特征空间）

        # 加载数据
        if data_path:
            self.load_raw_data(data_path)

    def load_raw_data(self, data_path: str) -> None:
        """加载CSV格式的真菌原始数据"""
        try:
            self.raw_data = pd.read_csv(data_path, encoding='utf-8')
            print(f"✅ 成功加载原始数据：{self.raw_data.shape[0]}行 × {self.raw_data.shape[1]}列")
        except FileNotFoundError:
            print(f"❌ 错误：未找到文件 {data_path}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载数据时出错: {str(e)}")
        
    def clean_data(self) -> None:
        """数据清洗：填充缺失值，过滤异常值"""
        if self.raw_data is None:
            raise ValueError("❌ 请先调用 load_raw_data 方法加载原始数据")
        
        self.cleaned_data = fill_missing_data(self.raw_data)   # 填充缺失值
        # print(f"✅ 数据清洗完成：{self.cleaned_data.shape[0]}行 × {self.cleaned_data.shape[1]}列")

        # 过滤异常值(3σ原则)
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = self.cleaned_data[col].mean()
            std = self.cleaned_data[col].std()
            # 过滤掉超出3σ范围的异常值
            self.cleaned_data = self.cleaned_data[
                (self.cleaned_data[col] >= mean - 3 * std) & 
                (self.cleaned_data[col] <= mean + 3 * std)
            ]
        # print(f"✅ 数据清洗完成：{self.cleaned_data.shape[0]}行 × {self.cleaned_data.shape[1]}列")

        # 重置索引
        self.cleaned_data = self.cleaned_data.reset_index(drop=True)
        print(f"✅ 数据清洗完成：{self.cleaned_data.shape[0]}行有效数据")

    def build_feature_space(
        self,
        core_features: list = None,
        norm_method: str = "zscore"
    ) -> np.ndarray:
        """
        构建真菌特征空间（后续聚类/分类的输入）

        :param self: 类实例本身
        :param core_features: 核心特征列列表
        :type core_features: list
        :param norm_method: 标准化方法（"zscore" 或 "minmax"）
        :type norm_method: str
        :return: 标准化后的特征空间矩阵(n_fungus x n_features)
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        # 默认核心特征
        if core_features is None:
            core_features = [
                # 分解率
                "p_10", "p_16", "p_22",
                # 菌丝延伸率
                "mu_10", "mu_16", "mu_22",
                # Q10系数
                "Q10",
                # 五种酶活性
                "enzyme_1", "enzyme_2", "enzyme_3", "enzyme_4", "enzyme_5"
            ]
        
        # 检查特征列是否存在
        if self.cleaned_data is None:
            self.clean_data()
        missing_cols = [col for col in core_features if col not in self.cleaned_data.columns]
        if missing_cols:
            raise ValueError(f"❌ 缺少核心特征列: {missing_cols}")
        
        # 提取核心特征并标准化
        self.feature_space = self.cleaned_data[core_features].values
        self.feature_space = normalize_feature(self.feature_space, method=norm_method)

        # 保存真菌 ID/名称 列表（用于后续映射）
        self.fungus_ids = self.cleaned_data["fungus_id"].values if "fungus_id" in self.cleaned_data.columns else np.arange(len(self.feature_space))

        print(f"✅ 特征空间构建完成：{self.feature_space.shape[0]}个真菌 × {self.feature_space.shape[1]}个特征")
        return self.feature_space
    
# 单元测试
if __name__ == "__main__":
    # ---------- 测试用例 ----------
    # 生成模拟数据（替代真实CSV）
    mock_data = pd.DataFrame({
        "fungus_id": [f"F{i}" for i in range(1, 35)],  # 34种真菌
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

    # 插入少量缺失值（模拟真实数据情况）
    mock_data.loc[5, "p_22"] = np.nan
    mock_data.loc[10, "mu_16"] = np.nan
    mock_data.loc[10, "enzyme_3"] = np.nan

    # 测试数据加载器
    loader = FungalDataLoader()
    loader.raw_data = mock_data  # 直接赋值模拟加载数据
    loader.clean_data()
    feature_space = loader.build_feature_space()

    # 输出结果检查
    print("\n========== 测试结果 ==========")
    print(f"特征空间形状: {feature_space.shape}")
    print(f"前5个真菌ID: {loader.fungus_ids[:5]}")
    print(f"前两个真菌的特征向量:\n{feature_space[:2]}")