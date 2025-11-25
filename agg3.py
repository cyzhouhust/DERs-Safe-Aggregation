import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from data import *
from matrices import build_matrices

# --- VPP 配置 ---
vpp_nodes = [10, 15, 18, 20, 25]
num_vpp = len(vpp_nodes) 
num_vars = 2 * num_vpp
T = 24  # 假设的时间步长

# I. 模拟 24 小时平滑动态 DER 边界 (请替换为您的真实数据)
BASE_P_MAX = np.array([40000, 10000, 40000, 4000, 4000])
# 基础最小容量（吸收）
BASE_P_MIN_ABS = np.array([1000,1000, 1000,1000,1000]) 

def generate_24h_der_limits(t):
    """
    根据时间步 t 模拟平滑的 DER 动态边界，防止剧烈变动导致 LP 不可行。
    t: 0 to 23
    """
    hour = t + 1
    scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi) 
    # P_inj_max: 最大输出功率 (约定 A: 正值)
    P_inj_max_t = BASE_P_MAX * scale
    # P_inj_min: 最小输出功率 (最大吸收能力，约定 A: 负值)
    # 吸收能力随着输出增加而略微增加
    P_inj_min_t = -BASE_P_MIN_ABS * (1 + 0.5 * scale)
    # Q 边界 (根据 P 动态调整，功率因素限制)
    Q_limit_t = P_inj_max_t * 0.5
    Q_inj_max_t = np.minimum(Q_limit_t, [250, 300, 200, 200, 350])
    Q_inj_min_t = -Q_inj_max_t
    return P_inj_min_t, P_inj_max_t, Q_inj_min_t, Q_inj_max_t

# --- 存储结果的数组 ---
max_network_injection = np.zeros(T)
max_network_absorption = np.zeros(T)
max_physical_injection_24h = np.zeros(T)
max_physical_absorption_24h = np.zeros(T)

# --- 目标函数 (只需一次定义) ---
c_inj = np.hstack([np.ones(num_vpp), np.zeros(num_vpp)])   # Min sum(P_B) -> Max Injection
c_abs = np.hstack([-np.ones(num_vpp), np.zeros(num_vpp)]) # Min sum(-P_B) -> Max Absorption

# II. 24 小时循环求解 (优化后的最大聚合容量)
print(f"--- 2. 正在进行 24 小时滚动优化 ({T} 个时间步) ---")

for t in range(T):
    # --- A. 获取当前时刻的动态 DER 边界 ---
    P_inj_min_t, P_inj_max_t, Q_inj_min_t, Q_inj_max_t = generate_24h_der_limits(t)
    # --- B. 计算当前时刻的物理容量 ---
    max_physical_injection_24h[t] = np.sum(P_inj_max_t)
    max_physical_absorption_24h[t] = -np.sum(P_inj_min_t)
    # --- C. 转换边界到约定 B (用于 LP 求解) ---
    P_B_min_list_t = -P_inj_max_t
    P_B_max_list_t = -P_inj_min_t
    bounds_P_B_t = list(zip(P_B_min_list_t, P_B_max_list_t))
    bounds_Q_B_t = list(zip(Q_inj_min_t, Q_inj_max_t))
    bounds_total_B_t = bounds_P_B_t + bounds_Q_B_t
    # --- D. 获取约束矩阵 ---
    try:
        A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
    except NameError:
        max_network_injection[:] = np.nan
        max_network_absorption[:] = np.nan
        print("错误: 无法执行 build_matrices。请检查依赖。")
        break
    
    # 组合和转换 A 矩阵到约定 B
    A_total_A = np.vstack([A_V, A_I]) 
    b_total_A = np.vstack([b_V, b_I]).flatten() 
    A_P_A_part = A_total_A[:, :num_vpp] 
    A_Q_A_part = A_total_A[:, num_vpp:] 
    A_P_B_part = -A_P_A_part
    A_total_B = np.hstack([A_P_B_part, A_Q_A_part]) 
    b_total_B = b_total_A

    # --- E. 求解 1: 最大注入功率 (上调能力) ---
    res_inj = linprog(c_inj, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total_B_t, method='highs')
    max_network_injection[t] = -res_inj.fun if res_inj.success else np.nan
    # --- F. 求解 2: 最大吸收功率 (下调能力) ---
    res_abs = linprog(c_abs, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total_B_t, method='highs')
    max_network_absorption[t] = res_abs.fun if res_abs.success else np.nan 
print("--- 优化完成 ---")

# III. 可视化
hours = np.arange(1, T + 1)
plt.figure(figsize=(12, 7))
# --- 1. 物理上限/下限 (动态曲线) ---
plt.plot(hours, max_physical_injection_24h, 'r--', linewidth=1.5, alpha=0.6, label='Physical Max Injection Limit (Dynamic Sum)')
plt.plot(hours, -max_physical_absorption_24h, 'b--', linewidth=1.5, alpha=0.6, label='Physical Max Absorption Limit (Dynamic Sum)')
# --- 2. 优化后的最大聚合容量 ---
plt.plot(hours, max_network_injection, 'r-o', linewidth=2, label='Network Constrained Max Injection')
plt.plot(hours, max_network_absorption, 'b-x', linewidth=2, label='Network Constrained Max Absorption')

# 填充可行区域 (Flexibility Region)
plt.fill_between(hours, -max_network_absorption, max_network_injection, color='gray', alpha=0.1, label='VPP Feasibility Region')

plt.axhline(0, color='k', linestyle=':', linewidth=1) # 零线

plt.xlabel('Time (Hour)')
plt.ylabel(r'VPP Aggregated Power ($P^{\mathrm{vpp}}$) (kW) [Injection(+), Absorption(-)]')
plt.title('24-Hour VPP Flexibility Range vs. Physical Limits (Smooth Dynamic Limits)')
plt.xticks(hours)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

print("\n--- 24 小时 VPP 聚合能力 ---")
print(f"动态物理注入上限 (kW): {max_physical_injection_24h}")
print(f"动态网络注入上限 (kW): {max_network_injection}")