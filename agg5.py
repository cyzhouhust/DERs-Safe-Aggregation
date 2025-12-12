import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ==========================================
# 1. 模拟环境与配置 (集成输入)
# ==========================================

# 资源分组配置
RESOURCE_GROUPS = {
    "Type_A (Node 10,15)": [0, 1], 
    "Type_B (Node 18)":    [2],    
    "Type_C (Node 20)":    [3],    
    "Type_D (Node 25)":    [4]     
}

# 物理参数基准
BASE_P_MAX = np.array([40000, 10000, 40000, 4000, 4000])     # 发电上限
BASE_P_MIN_ABS = np.array([1000, 1000, 1000, 1000, 1000])    # 吸收上限

def build_mock_matrices(num_vars):
    """
    模拟生成电网灵敏度矩阵 (A x <= b)
    这里生成一个'虚拟'电网约束，强制总功率不能超过物理极限的 80%，
    以此模拟电网阻塞，确保 实际容量 < 理论容量，方便观察效果。
    """
    # 模拟一个限制总功率的约束: sum(P) <= Limit
    # 变量前一半是 P，后一半是 Q
    n_p = num_vars // 2
    
    # 构造 A矩阵: [1, 1, 1, 1, 1, 0, 0...] (只限制 P 的总和)
    A_mock = np.zeros((2, num_vars))
    A_mock[0, :n_p] = 1.0   # 上限约束系数
    A_mock[1, :n_p] = -1.0  # 下限约束系数
    
    # 构造 b向量: 限制为物理总量的 80%
    phy_sum = np.sum(BASE_P_MAX)
    b_mock = np.array([phy_sum * 0.8, phy_sum * 0.8]) 
    
    return A_mock, b_mock

def get_physical_bounds(t):
    """计算 t 时刻的物理边界"""
    hour = t + 1
    # 模拟正弦波动
    scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi)
    
    # 理论最大注入 (Injection Max)
    p_inj_max = BASE_P_MAX * scale
    # 理论最大吸收 (Absorption Max, 这里用正值表示容量)
    p_abs_max = BASE_P_MIN_ABS * (1 + 0.2 * scale)
    
    # 无功边界
    q_lim = p_inj_max * 0.5
    return p_inj_max, p_abs_max, -q_lim, q_lim

# ==========================================
# 2. 核心计算逻辑
# ==========================================

def solve_snapshot_capacity(t_snapshot=12):
    num_vpp = 5
    num_vars = 2 * num_vpp
    
    # 1. 获取物理边界
    p_phys_inj, p_phys_abs, q_min, q_max = get_physical_bounds(t_snapshot)
    
    # --- 计算指标 A: 理论可调容量 (Theoretical) ---
    theo_inj_total = np.sum(p_phys_inj)
    theo_abs_total = np.sum(p_phys_abs)
    
    # 2. 获取约束矩阵 (模拟)
    A_ub, b_ub = build_mock_matrices(num_vars)
    
    # 3. 构造变量边界 Bounds (P_load 约定)
    # P_load = -P_inj
    # Range: [-Max_Inj, Max_Abs]
    bounds_P = list(zip(-p_phys_inj, p_phys_abs))
    bounds_Q = list(zip(q_min, q_max))
    bounds = bounds_P + bounds_Q
    
    # 4. 定义结果容器
    result_data = {
        'Up': {},   # 上调/注入
        'Down': {}  # 下调/吸收
    }
    
    # ==========================
    # 场景 1: 计算实际最大注入 (Actual Injection)
    # ==========================
    # Max Injection => Min Sum(P_load)
    c_inj = np.hstack([np.ones(num_vpp), np.zeros(num_vpp)])
    res_inj = linprog(c_inj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res_inj.success:
        actual_inj_total = -res_inj.fun
        p_vals = -res_inj.x[:num_vpp] # 提取 P 并转为正值
        
        # 统计分项
        result_data['Up']['Total'] = {
            'Theoretical': theo_inj_total,
            'Actual': actual_inj_total
        }
        
        for name, idxs in RESOURCE_GROUPS.items():
            # 数量
            count = len(idxs)
            # 理论 (该组物理边界之和)
            th_val = np.sum(p_phys_inj[idxs])
            # 实际 (优化结果之和)
            act_val = np.sum(p_vals[idxs])
            # 占比
            ratio = (act_val / actual_inj_total * 100) if actual_inj_total > 1e-3 else 0
            
            result_data['Up'][name] = {
                'Count': count,
                'Theoretical': th_val,
                'Actual': act_val,
                'Ratio': ratio
            }
            
    # ==========================
    # 场景 2: 计算实际最大吸收 (Actual Absorption)
    # ==========================
    # Max Absorption => Min Sum(-P_load)
    c_abs = np.hstack([-np.ones(num_vpp), np.zeros(num_vpp)])
    res_abs = linprog(c_abs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res_abs.success:
        actual_abs_total = res_abs.fun
        p_vals = res_abs.x[:num_vpp] # 正值即为吸收量
        
        result_data['Down']['Total'] = {
            'Theoretical': theo_abs_total,
            'Actual': actual_abs_total
        }
        
        for name, idxs in RESOURCE_GROUPS.items():
            count = len(idxs)
            th_val = np.sum(p_phys_abs[idxs])
            act_val = np.sum(p_vals[idxs])
            ratio = (act_val / actual_abs_total * 100) if actual_abs_total > 1e-3 else 0
            
            result_data['Down'][name] = {
                'Count': count,
                'Theoretical': th_val,
                'Actual': act_val,
                'Ratio': ratio
            }
            
    return result_data

# ==========================================
# 3. 输出展示 (Output)
# ==========================================

def print_clean_report(data, direction_name, unit='kW'):
    """格式化打印表格"""
    if not data: return
    
    # 转换为 DataFrame
    rows = []
    # 遍历每类资源
    for name in RESOURCE_GROUPS.keys():
        info = data.get(name, {})
        rows.append({
            'Resource Group': name,
            'Quantity': info.get('Count', 0),
            f'Theo Cap ({unit})': info.get('Theoretical', 0.0),
            f'Actual Cap ({unit})': info.get('Actual', 0.0),
            'Contrib (%)': info.get('Ratio', 0.0)
        })
    
    df = pd.DataFrame(rows)
    
    # 设置显示格式
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    # 获取总量数据
    total_info = data.get('Total', {})
    total_theo = total_info.get('Theoretical', 0)
    total_act = total_info.get('Actual', 0)
    
    print(f"\n>>> {direction_name} Capacity Report (Time: 12:00) <<<")
    print("-" * 85)
    print(df.to_string(index=False))
    print("-" * 85)
    print(f"SYSTEM TOTAL | {'--':<8} | {total_theo:>14.2f} | {total_act:>14.2f} | 100.00")
    print("=" * 85)

if __name__ == "__main__":
    # 执行计算
    results = solve_snapshot_capacity(t_snapshot=12)
    
    # 输出 1: 上调/注入 (Injection)
    print_clean_report(results['Up'], "MAX INJECTION (Up-Regulation)")
    
    # 输出 2: 下调/吸收 (Absorption)
    print_clean_report(results['Down'], "MAX ABSORPTION (Down-Regulation)")