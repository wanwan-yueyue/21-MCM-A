# utils/metrics.py
# 通用基础计算函数模块

import numpy as np
import pandas as pd

def q10_correction(
    value_ref:float,
    q10:float,
    t_ref:float = 22.0,
    t_current:float = None
) -> float:
    """
    Q10温度修正函数 -- 用于根据温度调整生物过程速率
    
    :param value_ref: 参考温度t_ref下的过程速率（如分解系数α_ref、生长速率μ_ref）
    :type value_ref: float
    :param q10: Q10系数，表示温度每升高10摄氏度，过程速率增加的倍数(通常在1.5~3.0之间)
    :type q10: float
    :param t_ref: 参考温度，默认为22摄氏度
    :type t_ref: float
    :param t_current: 当前温度，若为None则不进行修正
    :type t_current: float or None
    :return: 当前温度下修正后的过程速率
    :rtype: float
    """
    if t_current is None or t_current == t_ref:
        return value_ref
    else:
        temp_diff = t_current - t_ref       # 计算温度差
        correction_factor = q10 ** (temp_diff / 10.0)       # 计算修正因子
        return value_ref * correction_factor        # 返回修正后的值
    
def normalize_feature(
    feature_array: np.ndarray,
    axis: int = 0,
    method: str = 'zscore'
)-> np.ndarray:
    """
    特征标准化函数(用于真菌功能群聚类) -- 支持Z-score标准化和Min-Max归一化
    
    :param feature_array: 待标准化的特征矩阵(n_samples x n_features)
    :type feature_array: np.ndarray
    :param axis: 标准化轴，0表示按列归一化，1表示按行归一化
    :type axis: int
    :param method: 标注化方法，'zscore'表示Z-score标准化，'minmax'表示Min-Max归一化
    :type method: str
    :return: 标准化后的特征矩阵
    :rtype: np.ndarray
    """
    # 校验输入
    if method not in ['zscore', 'minmax']:
        raise ValueError("method参数必须为'zscore'或'minmax'")
    
    # Z-score标准化
    if method == 'zscore':
        mean = np.mean(feature_array, axis=axis, keepdims=True)
        std = np.std(feature_array, axis=axis, keepdims=True)
        normalized_array = (feature_array - mean) / (std + 1e-8)  # 避免除以零
        return normalized_array
    
    # Min-Max归一化
    elif method == 'minmax':
        min_val = np.min(feature_array, axis=axis, keepdims=True)
        max_val = np.max(feature_array, axis=axis, keepdims=True)
        normalized_array = (feature_array - min_val) / (max_val - min_val + 1e-8)  # 避免除以零
        return normalized_array
    
def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值填充
    
    :param df: 原始的DataFrame，可能包含缺失值
    :type df: pd.DataFrame
    :return: 填充缺失值后的DataFrame
    :rtype: DataFrame
    """
    # 数值型列使用均值填充，分类型列使用众数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0]) 

    return df

# 单元测试
if __name__ == "__main__":
    ## Q10温度修正函数单元测试
    print("=== Q10温度修正函数单元测试 ===")
    alpha_ref = 0.5     # 参考分解系数
    q10 = 2.0           # Q10系数
    t_current = 32.0    # 当前温度
    alpha_corrected = q10_correction(alpha_ref, q10, t_ref=22.0, t_current=t_current)
    print(f"参考分解系数: {alpha_ref}, 当前温度: {t_current}°C；\
          修正前分解系数：{alpha_ref}，修正后分解系数: {alpha_corrected:.4f}(理论：1.0)")
    
    ## 特征标准化函数单元测试
    print("\n=== 特征标准化函数单元测试 ===")
    feature_matrix = np.array([[1,10],[2,20],[3,30]])
    zscore_normalized = normalize_feature(feature_matrix, axis=0, method='zscore')
    minmax_normalized = normalize_feature(feature_matrix, axis=0, method='minmax')
    print("原始特征矩阵:\n", feature_matrix)
    print("Z-score标准化结果:\n", zscore_normalized)
    print("Min-Max归一化结果:\n", minmax_normalized)