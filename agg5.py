import numpy as np
from scipy.optimize import linprog
import pandas as pd # 用 pandas 展示表格更直观
# 假设 build_matrices 在当前文件或引用中
# from matrices import build_matrices 
from data import * # --- VPP 基础配置 ---
# 假设 build_matrices 来自你的 matrices.py
# 如果 matrices 和 data 模块在同一目录下，直接引用即可
try:
    from matrices import build_matrices
except ImportError:
    # 仅作占位，防止缺少文件报错，实际使用时请确保环境正确
    def build_matrices(verbose=False):
        raise ImportError("缺少 matrices.py 或 build_matrices 函数")
    
# 索引对应关系: 0->Node10, 1->Node15, 2->Node18, 3->Node20, 4->Node25
vpp_nodes = [10, 15, 18, 20, 25] 
num_vpp = len(vpp_nodes)

# 定义资源分组 (索引列表)
resource_groups = {
    "Type_A (Node 10,15)": [0, 1], # 前两个资源是同一类
    "Type_B (Node 18)":    [2],    # 第三个是独立类
    "Type_C (Node 20)":    [3],    # 第四个是独立类
    "Type_D (Node 25)":    [4]     # 第五个是独立类
}

# 物理边界配置 (同前)
BASE_P_MAX = np.array([40000, 10000, 40000, 4000, 4000])
BASE_P_MIN_ABS = np.array([1000, 1000, 1000, 1000, 1000])

def get_snapshot_limits(t):
    """获取单时间点的物理边界 (沿用之前的逻辑)"""
    hour = t + 1
    scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi)
    P_inj_max = BASE_P_MAX * scale
    P_inj_min = -BASE_P_MIN_ABS * (1 + 0.5 * scale)
    Q_limit = P_inj_max * 0.5
    Q_inj_max = np.minimum(Q_limit, [250, 300, 200, 200, 350])
    Q_inj_min = -Q_inj_max
    return P_inj_min, P_inj_max, Q_inj_min, Q_inj_max

def solve_single_snapshot(t_snapshot=12):
    """
    计算指定时间点 t 的上下调容量，并分析资源贡献占比。
    """
    print(f"\n====== 正在计算时刻 t={t_snapshot} (Hour {t_snapshot+1}) 的聚合能力与贡献 ======")

    # 1. 获取物理边界
    P_min, P_max, Q_min, Q_max = get_snapshot_limits(t_snapshot)
    
    # 2. 构建约束矩阵 (A_V, A_I 等)
    try:
        # 注意：这里假设 build_matrices 内部读取的负荷数据对应当前时刻
        # 如果 bus_Pd 是全天数据，需要在 build_matrices 内部修改取值逻辑
        A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
    except NameError:
        print("错误: 缺少 build_matrices 函数或数据依赖。")
        return

    # 3. 组装优化矩阵 (对应约束 Ax <= b)
    # 约定: 优化变量 x = [P_1...P_n, Q_1...Q_n]
    # 原始矩阵是 A * [P, Q]^T <= b (P, Q 为注入功率)
    # 线性规划通用形式 A_ub * x <= b_ub
    
    A_total = np.vstack([A_V, A_I])
    b_total = np.vstack([b_V, b_I]).flatten()
    
    # 这里的矩阵系数需要注意符号：
    # build_matrices 生成的是针对 "注入(Injection)" 的正向灵敏度
    # 优化变量 P_B (LP变量) 在注入场景下即为 P_inj，在吸收场景下也是 P (负值)
    # 所以直接使用 A_total 即可，无需取反，关键在于 bounds 的设置
    
    # 分割系数矩阵以便后续可能的操作 (这里直接用 A_total 也可以)
    A_P_part = A_total[:, :num_vpp]
    A_Q_part = A_total[:, num_vpp:]
    A_ub = np.hstack([A_P_part, A_Q_part]) # 保持原样

    # 设置变量边界 [(P_min, P_max), ..., (Q_min, Q_max), ...]
    bounds_P = list(zip(P_min, P_max))
    bounds_Q = list(zip(Q_min, Q_max))
    bounds_total = bounds_P + bounds_Q

    # --- 场景 1: 最大上调能力 (Max Injection) ---
    # 目标: max sum(P)  =>  min sum(-1 * P)
    c_inj = np.hstack([-np.ones(num_vpp), np.zeros(num_vpp)]) 
    
    res_inj = linprog(c_inj, A_ub=A_ub, b_ub=b_total, bounds=bounds_total, method='highs')
    
    if res_inj.success:
        total_inj_capacity = -res_inj.fun # 还原为正值
        P_dispatch_inj = res_inj.x[:num_vpp] # 提取最优解中的 P 部分
        
        # 计算贡献占比
        inj_contrib = {}
        print("\n--- [上调/注入] 优化结果 ---")
        print(f"系统最大可注入总功率: {total_inj_capacity:.4f} kW")
        for name, idx_list in resource_groups.items():
            group_power = np.sum(P_dispatch_inj[idx_list])
            ratio = (group_power / total_inj_capacity) * 100 if total_inj_capacity > 1e-3 else 0
            inj_contrib[name] = {'Power_kW': group_power, 'Ratio_pct': ratio}
            print(f"  > {name}: {group_power:.2f} kW ({ratio:.2f}%)")
    else:
        print("上调优化求解失败！")
        total_inj_capacity = 0
        inj_contrib = {}

    # --- 场景 2: 最大下调能力 (Max Absorption) ---
    # 目标: max sum(-P) => min sum(P) (因为 P 本身是负数，求和最小即绝对值最大)
    # 或者理解为: min sum(P_vars) 其中 P_vars 允许为负
    c_abs = np.hstack([np.ones(num_vpp), np.zeros(num_vpp)]) 
    
    res_abs = linprog(c_abs, A_ub=A_ub, b_ub=b_total, bounds=bounds_total, method='highs')

    if res_abs.success:
        # res_abs.fun 是 sum(P)，通常是负数。
        # 下调容量通常表述为 "可吸收多少"，取绝对值展示
        total_abs_capacity = res_abs.fun 
        P_dispatch_abs = res_abs.x[:num_vpp]
        
        abs_contrib = {}
        print("\n--- [下调/吸收] 优化结果 ---")
        print(f"系统最大可吸收总功率: {total_abs_capacity:.4f} kW (负值表示吸收)")
        for name, idx_list in resource_groups.items():
            group_power = np.sum(P_dispatch_abs[idx_list])
            # 占比计算：基于绝对值计算贡献比例
            ratio = (abs_contrib_val := abs(group_power) / abs(total_abs_capacity)) * 100 if abs(total_abs_capacity) > 1e-3 else 0
            abs_contrib[name] = {'Power_kW': group_power, 'Ratio_pct': ratio}
            print(f"  > {name}: {group_power:.2f} kW ({ratio:.2f}%)")
    else:
        print("下调优化求解失败！")
        total_abs_capacity = 0
        abs_contrib = {}

    return {
        'time': t_snapshot,
        'capacity': {'up': total_inj_capacity, 'down': total_abs_capacity},
        'up_contribution': inj_contrib,
        'down_contribution': abs_contrib
    }

if __name__ == "__main__":
    # 假设我们要计算中午 12 点的情况
    result = solve_single_snapshot(t_snapshot=12)
    
    # 简单的数据可视化展示
    if result:
        df_up = pd.DataFrame(result['up_contribution']).T
        df_down = pd.DataFrame(result['down_contribution']).T
        
        print("\n====== 最终报表 ======")
        print("1. 上调 (Injection) 资源贡献表:")
        print(df_up)
        print("\n2. 下调 (Absorption) 资源贡献表:")
        print(df_down)