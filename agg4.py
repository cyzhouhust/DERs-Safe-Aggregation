import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
# 假设 data.py 和 matrices.py 中的 build_matrices 函数可用
from data import *
from matrices import build_matrices

# --- VPP 配置 ---
vpp_nodes = [10, 15, 18, 20, 25]
num_vpp = len(vpp_nodes) 
num_vars = 2 * num_vpp
T = 24  # 假设的时间步长
PENALTY_ALPHA = 1e6 # 软约束惩罚权重，必须远大于功率量级

# =========================================================================
# I. 模拟 24 小时平滑动态 DER 边界 (请替换为您的真实数据)
# =========================================================================

# 基础最大容量 (kW)
BASE_P_MAX = np.array([40000, 10000, 40000, 4000, 4000])
BASE_P_MIN_ABS = np.array([-100, -50, -150, -80, -120]) 

def generate_24h_der_limits(t):
    """根据时间步 t 模拟平滑的 DER 动态边界。"""
    hour = t + 1
    scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi) 

    P_inj_max_t = BASE_P_MAX * scale
    P_inj_min_t = -BASE_P_MIN_ABS * (1 + 0.5 * scale)
    
    Q_limit_t = P_inj_max_t * 0.5
    Q_inj_max_t = np.minimum(Q_limit_t, [250, 300, 200, 200, 350])
    Q_inj_min_t = -Q_inj_max_t
    
    return P_inj_min_t, P_inj_max_t, Q_inj_min_t, Q_inj_max_t

# --- 存储结果的数组 ---
max_network_injection = np.zeros(T)
max_network_absorption = np.zeros(T)
max_physical_injection_24h = np.zeros(T)
max_physical_absorption_24h = np.zeros(T)

# =========================================================================
# II. 24 小时循环求解 (使用软约束)
# =========================================================================
print(f"--- 2. 正在进行 24 小时滚动优化 ({T} 个时间步) (软约束启用) ---")

for t in range(T):
    # --- A. 获取约束矩阵 ---
    try:
        A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
    except NameError:
        print("错误: 无法执行 build_matrices。")
        break
    
    # 组合原始 (约定 A) 约束
    A_total_A = np.vstack([A_V, A_I]) 
    b_total_A = np.vstack([b_V, b_I]).flatten() 
    
    # 转换为 约定 B
    A_P_A_part = A_total_A[:, :num_vpp] 
    A_Q_A_part = A_total_A[:, num_vpp:] 
    A_P_B_part = -A_P_A_part
    A_total_B = np.hstack([A_P_B_part, A_Q_A_part]) 
    b_total_B = b_total_A
    
    num_constraints = len(b_total_B)
    
    # --- B. 构建软约束的增强矩阵 A_aug 和 b_aug ---
    
    # 1. 约束矩阵增强： [A_total_B, -I]
    I_slack = -np.eye(num_constraints)
    A_aug = np.hstack([A_total_B, I_slack]) # 新增 M 列，对应 slack 变量 S
    
    # 2. 边界约束增强 (仅对 Slack 变量): S >= 0
    # P_B/Q_B 边界
    P_inj_min_t, P_inj_max_t, Q_inj_min_t, Q_inj_max_t = generate_24h_der_limits(t)
    P_B_min_list_t = -P_inj_max_t
    P_B_max_list_t = -P_inj_min_t
    bounds_P_B_t = list(zip(P_B_min_list_t, P_B_max_list_t))
    bounds_Q_B_t = list(zip(Q_inj_min_t, Q_inj_max_t))
    
    # Slack 边界： S >= 0
    bounds_S = [(0, None)] * num_constraints
    bounds_total_aug = bounds_P_B_t + bounds_Q_B_t + bounds_S
    
    # 3. 目标函数增强： [c_orig, alpha * 1_M]
    c_slack = np.full(num_constraints, PENALTY_ALPHA)

    # --- C. 计算物理容量 (用于绘图) ---
    max_physical_injection_24h[t] = np.sum(P_inj_max_t)
    max_physical_absorption_24h[t] = -np.sum(P_inj_min_t)

    # --- D. 求解 1: 最大注入功率 (Min sum(P_B) + alpha sum(S)) ---
    c_inj_aug = np.hstack([np.ones(num_vpp), np.zeros(num_vpp), c_slack])
    res_inj = linprog(c_inj_aug, A_ub=A_aug, b_ub=b_total_B, bounds=bounds_total_aug, method='highs')
    
    if res_inj.success:
        # 最大注入功率 = -Min(Sum P_B)
        P_B_opt = res_inj.x[:num_vpp]
        max_network_injection[t] = -np.sum(P_B_opt)
    else:
        # 即使使用了软约束，如果 DER 边界导致不可行，也可能失败，但概率大大降低
        max_network_injection[t] = np.nan
        
    # --- E. 求解 2: 最大吸收功率 (Min -sum(P_B) + alpha sum(S)) ---
    c_abs_aug = np.hstack([-np.ones(num_vpp), np.zeros(num_vpp), c_slack])
    res_abs = linprog(c_abs_aug, A_ub=A_aug, b_ub=b_total_B, bounds=bounds_total_aug, method='highs')

    if res_abs.success:
        # 最大吸收功率 = Max(Sum P_B)
        P_B_opt = res_abs.x[:num_vpp]
        max_network_absorption[t] = np.sum(P_B_opt)
    else:
        max_network_absorption[t] = np.nan
    
    # 可选：打印松弛量以监控约束违反程度
    if res_inj.success:
        slack_inj = res_inj.x[num_vars:]
        max_slack_inj = np.max(slack_inj)
        if max_slack_inj > 1e-4:
            print(f"时刻 {t+1} (注入): 警告！最大松弛量达到 {max_slack_inj:.2e}")
    
print("--- 优化完成 ---")

# =========================================================================
# III. 可视化
# =========================================================================

hours = np.arange(1, T + 1)
plt.figure(figsize=(12, 7))

# --- 1. 物理上限/下限 (动态曲线) ---
plt.plot(hours, max_physical_injection_24h, 'r--', linewidth=1.5, alpha=0.6, label='Physical Max Injection Limit (Dynamic Sum)')
plt.plot(hours, -max_physical_absorption_24h, 'b--', linewidth=1.5, alpha=0.6, label='Physical Max Absorption Limit (Dynamic Sum)')


# --- 2. 优化后的最大聚合容量 ---
plt.plot(hours, max_network_injection, 'r-o', linewidth=2, label='Network Constrained Max Injection')
plt.plot(hours, -max_network_absorption, 'b-x', linewidth=2, label='Network Constrained Max Absorption')

# 填充可行区域 (Flexibility Region)
plt.fill_between(hours, -max_network_absorption, max_network_injection, color='gray', alpha=0.1, label='VPP Feasibility Region')

plt.axhline(0, color='k', linestyle=':', linewidth=1) # 零线

plt.xlabel('Time (Hour)')
plt.ylabel(r'VPP Aggregated Power ($P^{\mathrm{vpp}}$) (kW) [Injection(+), Absorption(-)]')
plt.title('24-Hour VPP Flexibility Range vs. Physical Limits (Soft Constraints)')
plt.xticks(hours)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

print("\n--- 24 小时 VPP 聚合能力 ---")
print(f"动态物理注入上限 (kW): {max_physical_injection_24h}")
print(f"动态网络注入上限 (kW): {max_network_injection}")