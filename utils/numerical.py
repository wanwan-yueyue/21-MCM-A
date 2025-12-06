# utils/numerical.py
# 计算数值积分

import numpy as np

def rk4(initial_state: np.ndarray, derivative_func, dt: float, *args) -> np.ndarray:
    """
    四阶龙格-库塔法(rk4) -- 用于求解常微分方程的数值方法 
    
    :param initial_state: 初始状态向量（如[木质纤维质量W, 真菌1生物量F1, 真菌2生物量F2]）
    :type initial_state: np.ndarray
    :param derivative_func: 计算状态导数的函数，接受状态向量和其他参数，返回状态向量的导数。格式：dstate/dt = func(state, *args)
    :param dt: 时间步长（如1天）
    :type dt: float
    :param args: 传递给derivative_func的其他参数（如温度T、湿度M、模型参数）
    :return: 积分后的新状态向量
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """

    #校验输入
    if not isinstance(initial_state, np.ndarray):
        raise ValueError("initial_state必须是一个numpy数组")
    if dt <= 0:
        raise ValueError("时间步长dt必须为正数")
    
    # 计算RK4的四个斜率
    k1 = derivative_func(initial_state, *args)
    k2 = derivative_func(initial_state + 0.5 * dt * k1, *args)
    k3 = derivative_func(initial_state + 0.5 * dt * k2, *args)
    k4 = derivative_func(initial_state + dt * k3, *args)

    # 加权更新状态
    new_state = initial_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

# 单元测试：验证RK4函数的正确性
if __name__ == "__main__":
    # 测试用例：dy/dt = -y, y(0) = 1, 解析解y(t) = exp(-t)
    def test_derivative(state: np.ndarray, *args) -> np.ndarray:
        return -1 * state

    # 初始条件
    initial_y = np.array([1.0])     # 初始值y(0) = 1
    dt = 0.1
    local_time = 1.0
    steps = int(local_time / dt)

    # 使用RK4进行逐步积分
    current_state = initial_y.copy()
    for _ in range(steps):
        current_state = rk4(current_state, test_derivative, dt)

     # 验证结果（理论值≈0.3679）
    print("=== RK4 单元测试 ===")
    print(f"RK4计算结果 y(1.0) = {current_state[0]:.4f}")
    print(f"解析解 y(1.0) = {np.exp(-1):.4f}")
    print(f"误差：{abs(current_state[0] - np.exp(-1)):.6f}")  # 误差应<1e-4