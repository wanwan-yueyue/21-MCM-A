# data/environment_db.py
# 环境参数 / 时间序列生成

# 导入项目根目录到sys.path，确保可以导入utils模块（快速调试用）
import sys
import os

# 获取当前脚本（data_loader.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（21-A-final/，即current_dir的上级目录）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path
sys.path.append(root_dir)

import numpy as np
from utils.metrics import q10_correction

class EnvironmentGenerator:
    """
    环境参数生成器
    功能：生成五类环境（干旱/半干旱/半湿润/湿润/潮湿）的温度湿度时间序列，含季节性+趋势
    """
    # 五类环境的基础参数
    BASE_ENV_PARAMS = {
        'arid':{ # 干旱
            'mu_M': 0.2,  # 平均湿度（0~1）
            'sigma_M': 0.1,  # 湿度波动标准差
            'mu_T': 28.0,   # 平均温度（°C）
            'sigma_T': 4.0,  # 温度波动标准差
            'season_amp_M': 0.05,  # 湿度季节性振幅
            'season_amp_T': 2.0,   # 温度季节性振
        },
        'semi_arid':{ # 半干旱
            'mu_M': 0.35,
            'sigma_M': 0.12,
            'mu_T': 25.0,
            'sigma_T': 3.5,
            'season_amp_M': 0.08,
            'season_amp_T': 1.8,
        },
        'semi_humid':{ # 半湿润
            'mu_M': 0.5,
            'sigma_M': 0.1,
            'mu_T': 22.0,
            'sigma_T': 3.0,
            'season_amp_M': 0.1,
            'season_amp_T': 1.5,
        },
        'humid':{ # 湿润
            'mu_M': 0.65,
            'sigma_M': 0.08,
            'mu_T': 20.0,
            'sigma_T': 2.5,
            'season_amp_M': 0.08,
            'season_amp_T': 1.2,
        },
        'wet':{ # 潮湿
            'mu_M': 0.8,
            'sigma_M': 0.05,
            'mu_T': 18.0,
            'sigma_T': 2.0,
            'season_amp_M': 0.05,
            'season_amp_T': 1.0,
        }
    }

    def __init__(
        self, 
        env_type: str = 'semi_humid', 
        duration_days: int = 365,
        climate_trend: str = 'None' # warming/cooling/None
    ):
        # 校验环境类型
        if env_type not in self.BASE_ENV_PARAMS:
            raise ValueError(f"❌ 未知环境类型: {env_type}\n \
                             请从 {list(self.BASE_ENV_PARAMS.keys())} 中选择。")
        
        self.env_type = env_type
        self.params = self.BASE_ENV_PARAMS[env_type]
        self.duration_days = duration_days
        self.climate_trend = climate_trend

        # 生成时间轴（天）
        self.timeline = np.arange(duration_days)
        # 初始化温度湿度序列
        self.M: np.ndarray = None  # 湿度时间序列（0~1）
        self.T: np.ndarray = None  # 温度时间序列（°C）

    def generate_environment_series(self) -> tuple[np.ndarray, np.ndarray]:
        """
        生成温度湿度时间序列(温度+湿度)：基础均值 + 季节性波动 + 随机噪声 + 气候变化趋势

        :return: (湿度序列M, 温度序列T)
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        # 基础均值
        base_M = self.params['mu_M']
        base_T = self.params['mu_T']

        # 季节性波动（正弦函数模拟年周期）
        # 湿度季节波动（相位0）
        seasonality_M = self.params['season_amp_M'] * np.sin(2 * np.pi * self.timeline / 365)
        # 温度季节波动（相位π/2，夏季最高）
        seasonality_T = self.params['season_amp_T'] * np.sin(2 * np.pi * self.timeline / 365 + np.pi/2)

        # 随机噪声
        noise_M = np.random.normal(0, self.params['sigma_M'], self.duration_days)
        noise_T = np.random.normal(0, self.params['sigma_T'], self.duration_days)

        # 合成基础序列
        self.M = base_M + seasonality_M + noise_M
        self.T = base_T + seasonality_T + noise_T

        # 应用气候变化趋势
        if self.climate_trend == 'warming':
            # 线性变暖：每天升温0.005°C（每年约1.825°C）
            self.T += 0.005 * self.timeline
        elif self.climate_trend == 'cooling':
            # 线性变冷：每天降温0.003°C
            self.T -= 0.003 * self.timeline

        # 限制湿度在0~1范围内，温度不低于0°C
        self.M = np.clip(self.M, 0.0, 1.0)
        self.T = np.clip(self.T, 0.0, None)

        print(f"✅ 环境序列生成完成：{self.env_type}环境，持续{self.duration_days}天")
        return self.M, self.T
    
    def get_env_params_at_t(self, t: int) -> dict:
        """
        获取某一时间点的环境参数，含Q10矫正的基础参数（湿度M，温度T）
        
        :param self: 类实例本身 
        :param t: 时间点（天）
        :type t: int
        :return: 环境参数字典
        :rtype: dict
        """

        if self.M is None or self.T is None:
            self.generate_environment_series()
        
        # 校验时间点
        if t < 0 or t >= self.duration_days:
            raise ValueError(f"❌ 时间点t={t}超出范围，应在0到{self.duration_days-1}之间。")
        
        # 返回当前时间点的环境参数
        return {
            'M': self.M[t],
            'T': self.T[t],
            'mu_M': self.params['mu_M'],
            'mu_T': self.params['mu_T'],
            'sigma_M': self.params['sigma_M'],
            'sigma_T': self.params['sigma_T'],
        }
    
# 单元测试
if __name__ == "__main__":
     # 测试：生成半湿润环境的365天序列（含变暖趋势）
    env_gen = EnvironmentGenerator(
        env_type="semi_humid",
        duration_days=365,
        climate_trend="warming"
    )
    M, T = env_gen.generate_environment_series()

    # 输出统计信息
    print("\n=== 环境序列测试结果 ===")
    print(f"环境类型：{env_gen.env_type}")
    print(f"湿度范围：{M.min():.2f} ~ {M.max():.2f}（均值：{M.mean():.2f}）")
    print(f"温度范围：{T.min():.2f} ~ {T.max():.2f}（均值：{T.mean():.2f}）")
    print(f"第180天（夏季）环境参数：{env_gen.get_env_params_at_t(180)}")

    # 可视化（可选，验证趋势）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(env_gen.timeline, M, label="Humidity", color="blue")
        plt.title(f"{env_gen.env_type} - Humidity time series")
        plt.xlabel("Days")
        plt.ylabel("Humidity(0~1)")
        plt.legend()

        plt.subplot(122)
        plt.plot(env_gen.timeline, T, label="Temperature", color="red")
        plt.title(f"{env_gen.env_type} - Temperature time series (warming trend)")
        plt.xlabel("Days")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("⚠️ matplotlib is not installed, skipping visualization")