# fungal_decomposition/main.py

import os
import sys
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（21-A-final/）
root_dir = os.path.dirname(current_dir)
# 将根目录加入sys.path
sys.path.append(root_dir)

import argparse
import matplotlib.pyplot as plt
import numpy as np

from simulation.simulator import FungalDecompositionSimulator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="木质纤维真菌分解仿真框架主程序")
    # 核心配置
    parser.add_argument("--data_path", type=str, default=None, help="真菌原始数据CSV路径（默认使用模拟数据）")
    parser.add_argument("--env_type", type=str, default="semi_humid", 
                        choices=["arid", "semi_arid", "semi_humid", "humid", "wet"],
                        help="环境类型（干旱/半干旱/半湿润/湿润/潮湿）")
    parser.add_argument("--duration_days", type=int, default=365, help="仿真时长（天）")
    parser.add_argument("--climate_trend", type=str, default=None, 
                        choices=["warming", "cooling", None],
                        help="气候变化趋势（变暖/变冷/无）")
    # 输出配置
    parser.add_argument("--save_path", type=str, default="simulation_results.csv", help="结果保存路径")
    parser.add_argument("--plot", action="store_true", help="是否绘制仿真结果可视化图")
    parser.add_argument("--calibrate", action="store_true", help="是否校准模型参数")
    
    return parser.parse_args()

def plot_results(simulator):
    """仿真结果可视化"""
    timeline = simulator.env_gen.timeline
    W_history = simulator.results["W_history"]
    T_history = simulator.results["env_history"]["T"]
    M_history = simulator.results["env_history"]["M"]
    F_history = simulator.results["F_history"]

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Fungal Decomposition Simulation Results", fontsize=14)

    # 木质纤维质量变化
    axes[0,0].plot(timeline, W_history, color="brown", label="Wood Mass")
    axes[0,0].set_xlabel("Days")
    axes[0,0].set_ylabel("Wood Mass (g)")
    axes[0,0].set_title("Wood Fiber Decomposition")
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    # 环境参数（温度+湿度）
    ax2 = axes[0,1]
    ax2_twin = ax2.twinx()
    ax2.plot(timeline, T_history, color="red", label="Temperature", alpha=0.7)
    ax2_twin.plot(timeline, M_history, color="blue", label="Humidity", alpha=0.7)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Temperature (℃)", color="red")
    ax2_twin.set_ylabel("Humidity (0~1)", color="blue")
    ax2.set_title("Environmental Parameters")
    ax2.tick_params(axis='y', labelcolor="red")
    ax2_twin.tick_params(axis='y', labelcolor="blue")
    ax2.grid(alpha=0.3)

    # 前3种真菌生物量变化
    axes[1,0].set_title("Fungal Biomass (Top 3)")
    axes[1,0].set_xlabel("Days")
    axes[1,0].set_ylabel("Biomass (g)")
    colors = ["green", "orange", "purple"]
    for i, (fid, vals) in enumerate(list(F_history.items())[:3]):
        # 补全长度（适配时间轴）
        vals = vals + [vals[-1]]*(len(timeline)-len(vals)) if len(vals)<len(timeline) else vals[:len(timeline)]
        axes[1,0].plot(timeline, vals, color=colors[i], label=f"Fungus {fid}")
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # 核心评估指标
    metrics = simulator.results["metrics"]
    metric_names = ["DEI", "Diversity", "Stability", "Synergy Gain"]
    metric_values = [metrics["DEI"], metrics["diversity_index"], metrics["stability"], metrics["synergy_gain"]]
    axes[1,1].bar(metric_names, metric_values, color=["darkgreen", "darkblue", "darkorange", "darkred"])
    axes[1,1].set_title("Core Evaluation Metrics")
    axes[1,1].set_ylabel("Value")
    # 标注数值
    for i, v in enumerate(metric_values):
        axes[1,1].text(i, v+0.01, f"{v:.2f}", ha="center")
    axes[1,1].grid(alpha=0.3, axis="y")

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("simulation_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    """主程序逻辑"""
    # 解析参数
    args = parse_args()
    print("=== 木质纤维真菌分解仿真框架 ===")
    print(f"配置参数：\n- 环境类型：{args.env_type}\n- 仿真时长：{args.duration_days}天\n- 气候变化：{args.climate_trend}\n- 参数校准：{args.calibrate}")

    # 初始化仿真器
    simulator = FungalDecompositionSimulator(
        data_path=args.data_path,
        env_type=args.env_type,
        duration_days=args.duration_days,
        climate_trend=args.climate_trend,
        calibrate_params=args.calibrate
    )

    # 运行仿真
    simulator.run_simulation()

    # 导出结果
    simulator.export_results(args.save_path)

    # 可视化（可选）
    if args.plot:
        print("=== 绘制仿真结果可视化图 ===")
        plot_results(simulator)

    # 输出核心结论
    print("\n=== 仿真核心结论 ===")
    print(f"1. 分解效率系数（DEI）：{simulator.results['metrics']['DEI']:.4f}")
    print(f"2. 真菌多样性指数：{simulator.results['metrics']['diversity_index']:.4f}")
    print(f"3. 分解稳定性指数：{simulator.results['metrics']['stability']:.4f}")
    print(f"4. 协同增益指数：{simulator.results['metrics']['synergy_gain']:.4f}")
    print(f"5. 最小生物多样性阈值：{simulator.results['min_diversity_threshold']:.2f}")
    print("\n=== 仿真完成 ===")

if __name__ == "__main__":
    # 确保结果保存目录存在
    if not os.path.exists("results"):
        os.makedirs("results")
    main()