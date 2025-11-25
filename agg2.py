import numpy as np
from scipy.optimize import linprog
from data import *
from matrices import build_matrices
# 假设 data.py 和 matrices.py 中的 build_matrices 函数可用

# --- VPP 配置 (必须与 build_matrices 中的 vpp_nodes 一致) ---
vpp_nodes = [10, 15, 18, 20, 25]
num_vpp = len(vpp_nodes) 
num_vars = 2 * num_vpp

# --- DER 边界配置 (基于 约定 A: 注入为正) ---
# 这代表了 VPP 资源的物理能力。
# P_inj_min < 0 表示最大吸收能力，P_inj_max > 0 表示最大输出能力。
P_inj_min_list = np.array([0, 0, 0, 0, 0])
P_inj_max_list = np.array([40000, 10000, 40000, 4000, 4000])
Q_inj_min_list = np.array([-250, -300, -200, -200, -350]) # 假设Q也有边界
Q_inj_max_list = np.array([250, 300, 200, 200, 350])

# =========================================================================
# I. 获取和转换约束矩阵 (从 约定 A 到 约定 B)
# =========================================================================
print("--- 1. 正在获取和转换约束矩阵 ---")

# 调用您的 build_matrices 函数，获取基于约定 A 的原始约束
try:
    A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
except NameError:
    print("错误: 无法执行 build_matrices。请确保 data.py 和 build_matrices 定义在当前环境中。")
    exit()

# 1. 组合原始 (约定 A) 约束
A_total_A = np.vstack([A_V, A_I]) # (128, 10)
b_total_A = np.vstack([b_V, b_I]).flatten() # (128,)

# 2. 转换为 约定 B (吸收为正, P_B = -P_A) 的约束矩阵
# 约束形式：A_P * P_A + A_Q * Q_A <= b
# 替换 P_A = -P_B：A_P * (-P_B) + A_Q * Q_B <= b
# 得到：(-A_P) * P_B + A_Q * Q_B <= b

A_P_A_part = A_total_A[:, :num_vpp] # A_total_A 的前 N 列是 P_A 的系数
A_Q_A_part = A_total_A[:, num_vpp:] # A_total_A 的后 N 列是 Q_A 的系数

# P 的系数矩阵取负
A_P_B_part = -A_P_A_part
A_Q_B_part = -A_Q_A_part
A_total_B = np.hstack([A_P_B_part, A_Q_A_part]) # 新的约束矩阵 (约定 B)
b_total_B = b_total_A

# 3. 转换为 约定 B 的边界 (P_B > 0 为吸收, P_B < 0 为注入)
# P_B_min = -P_A_max, P_B_max = -P_A_min
P_B_min_list = -P_inj_max_list
P_B_max_list = -P_inj_min_list

# Q 边界 (Q_B = Q_A)
Q_B_min_list = Q_inj_min_list
Q_B_max_list = Q_inj_max_list

bounds_P_B = list(zip(P_B_min_list, P_B_max_list))
bounds_Q_B = list(zip(Q_B_min_list, Q_B_max_list))
bounds_total_B = bounds_P_B + bounds_Q_B

# =========================================================================
# II. 优化求解 (最大注入功率)
# =========================================================================

# 目标: 最大化注入功率 (Max P_inj_total) 
#       等价于最小化 (Min P_total) = Min (sum P_B_j)
c_P = np.ones(num_vpp)  
c_Q = np.zeros(num_vpp) 
c_B = np.hstack([c_P, c_Q]) 

print("--- 2. 正在求解 VPP 最大注入功率 ---")
result = linprog(
    c_B,                       # 目标函数：最小化 sum(P_B_j)
    A_ub=A_total_B,            # 约束矩阵 (约定 B)
    b_ub=b_total_B,
    bounds=bounds_total_B,     # 边界 (约定 B)
    method='highs' 
)

# =========================================================================
# III. 结果分析
# =========================================================================
if result.success:
    print("\n✅ 优化成功！")
    
    X_opt = result.x
    P_B_opt = X_opt[:num_vpp]  # P_B: 吸收 (正) / 注入 (负)
    
    # Total_P_min (result.fun) 是一个负值
    Total_P_min = result.fun 
    Total_P_inj_max = -Total_P_min
    
    print("\n### 最终优化结果 (最大注入功率) ###")
    print(f"最小 VPP 总 P (Sum P_B): {Total_P_min:.2f} kW")
    print(f"✨ **最大 VPP 总注入功率** (P_inj, max): {Total_P_inj_max:.2f} kW")
    
    print("\n--- 最优 VPP 调度方案 ---")
    print(f"{'节点':<6} | {'P (值)':<10} | {'Q (值)':<10} | {'动作':<6}")
    print("-" * 38)
    for i, node in enumerate(vpp_nodes):
        P_val = P_B_opt[i]
        Q_val = X_opt[num_vpp + i]
        
        action = "吸收" if P_val > 1e-3 else ("注入" if P_val < -1e-3 else "不动作")
        P_display = abs(P_val)
        
        print(f"{node:<6} | {P_display:<10.2f} | {Q_val:<10.2f} | {action:<6}")
        
    # 检查约束紧度
    slack = b_total_B - A_total_B @ X_opt
    min_slack = np.min(slack)
    
    if min_slack < 1e-6:
        print(f"\n🚨 注意: 至少一个网络安全约束已触及边界（松弛量: {min_slack:.4e}）。")
    else:
        print(f"\n约束最紧点松弛量: {min_slack:.4e}")
        
else:
    print(f"\n❌ 优化失败: {result.message}")
    print("这可能意味着问题是不可行的，请检查 DER 边界是否过于严格。")