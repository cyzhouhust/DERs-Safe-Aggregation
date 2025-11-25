import numpy as np
from scipy.optimize import linprog
from data import *
from matrices import build_matrices

# 构建或获取约束与灵敏度矩阵
# build_matrices 现在返回 14 个值（包括诊断向量），全部接收以避免解包错误
A_V, b_V, A_I, b_I, R_hat, X_hat, R_I, X_I, Pd, Qd,C_MI_load_neg, C_MI_load_pos, C_mV_load, C_MV_load = build_matrices()
# 假设 VPP 节点集
vpp_nodes = [10, 15, 18, 20, 25]
vpp_indices = np.array(vpp_nodes) - 1 # 表示 0-based index: [9, 14, 17, 19, 24]
num_vpp = len(vpp_indices) #表示vpp管理的资源数量   

# 组合所有约束
#A_total = np.vstack([A_V, A_I]) # (64 + 64) x 10 = 128 x 10
#b_total = np.vstack([b_V, b_I]).flatten() # 128 x 1 -> 128
A_total = A_V
b_total = b_V

# 1. 定义目标函数系数 c
c_P = np.ones(num_vpp) # 目标函数中 P 的系数
c_Q = np.zeros(num_vpp) # 目标函数中 Q 的系数
c = np.hstack([c_P, c_Q]) 

# 2. 定义变量边界 (Bounds)
P_min_list = [-100, -600, -40000, -50000, -5000]
P_max_list = [0, 0, 4000, 4000, 2000]
#Q_min_list = [-250, -300, -200, -200, -350]
#Q_max_list = [250, 300, 200, 200, 350]
Q_min_list = [0, 0, 0, 0, 0]
Q_max_list = [0, 0, 0, 0, 0]

print("Q_min_list =", Q_min_list, type(Q_min_list))
print("Q_max_list =", Q_max_list, type(Q_max_list))

# P_vpp 的边界
bounds_P = list(zip(P_min_list, P_max_list))
# Q_vpp 的边界
bounds_Q = list(zip(Q_min_list, Q_max_list))

# 总边界 (10 个变量)
bounds_total = bounds_P + bounds_Q
# =========================================================================
# 3. 使用 linprog 求解

print("--- 正在求解 VPP 优化问题 ---")
try:
    result = linprog(
        c,
        A_ub=A_total,
        b_ub=b_total,
        bounds=bounds_total,
        method='highs' # 推荐使用 'highs' 算法
    )
except NameError:
    # 如果 A_total, b_total 未在当前环境中定义（因为我无法执行您的原始代码），
    # 打印一个占位符消息
    print("错误：约束矩阵 A_total 和 b_total 未在当前环境中定义。请确保先运行灵敏度计算部分。")
    result = None

# =========================================================================
# 4. 结果分析
if result and result.success:
    print("\n✅ 优化成功！")
    # 提取最优解
    X_opt = result.x
    P_vpp_opt = X_opt[:num_vpp]
    Q_vpp_opt = X_opt[num_vpp:]
    Total_P_max = -result.fun # 最大化 -Z 得到 Z 的最大值
    print("\n### 优化结果 ###")
    print(f"最大 VPP 总有功输出 (P_vpp, max): {Total_P_max:.2f} kW")
    print(f"最小化目标函数值 (-P_vpp): {result.fun:.2f}")
    print("\n--- 最优 VPP 调度方案 ---")
    for i, node in enumerate(vpp_nodes):
        print(f"节点 {node}: P_vpp = {P_vpp_opt[i]:.2f} kW, Q_vpp = {Q_vpp_opt[i]:.2f} kVar")
    print(f"\n约束满足状态 (松弛变量): \n{result.slack}")
elif result:
    print(f"\n❌ 优化失败: {result.message}")
    print("这可能意味着问题是不可行的（约束太严格）或无界的。")
else:
     print("\n❌ 无法执行 linprog，请检查前一步骤的约束矩阵是否正确生成。")