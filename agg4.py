import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ==========================================
# 1. 依赖导入
# ==========================================
try:
    from matrices import build_matrices
except ImportError:
    print("【严重错误】找不到 matrices.py 文件。")
    def build_matrices(verbose=False): return None

# ==========================================
# 2. VPP 聚合器核心类 (含修复逻辑)
# ==========================================
class VPPAggregator:
    def __init__(self, vpp_config):
        self.vpp_config = vpp_config
        # 节点顺序必须与矩阵列严格对应
        self.ordered_vpp_nodes = [10, 15, 18, 20, 25] 
        self.num_vpp_total = len(self.ordered_vpp_nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.ordered_vpp_nodes)}
        
        self.matrices_ready = False
        self._load_network_matrices()

    def _load_network_matrices(self):
        try:
            ret = build_matrices(verbose=False)
            if ret is None: return
            A_V, b_V, A_I, b_I = ret[0], ret[1], ret[2], ret[3]
            
            # 堆叠约束
            A_total_full = np.vstack([A_V, A_I])
            self.b_total = np.vstack([b_V, b_I]).flatten()
            
            # 维度适配
            current_cols = A_total_full.shape[1]
            if current_cols == 2 * self.num_vpp_total:
                self.A_total = A_total_full
            else:
                idx_p = [n - 1 for n in self.ordered_vpp_nodes]
                idx_q = [n - 1 + 33 for n in self.ordered_vpp_nodes] 
                self.A_total = A_total_full[:, idx_p + idx_q]

            # 检查并清理矩阵中的 NaN/Inf
            self.A_total = np.nan_to_num(self.A_total, nan=0.0)
            self.b_total = np.nan_to_num(self.b_total, nan=0.0)
            
            self.matrices_ready = True
            print(f"矩阵加载成功，形状: {self.A_total.shape}")
        except Exception as e:
            print(f"矩阵初始化失败: {e}")

    def _solve_robust(self, c, A_ub, b_ub, bounds):
        """
        鲁棒求解器：当硬约束无解时，自动降级为软约束求解
        """
        # 1. 尝试标准求解 (硬约束)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x[:self.num_vpp_total], -res.fun if c[0]<0 else res.fun
        
        # 2. 如果无解，启用软约束 (Soft Constraints)
        n_vars = len(c)
        n_constraints = len(b_ub)
        penalty_weight = 1e5 # 巨大的惩罚系数
        
        # 构造软约束问题
        c_soft = np.concatenate([c, np.ones(n_constraints) * penalty_weight])
        eye_neg = -np.eye(n_constraints)
        A_soft = np.hstack([A_ub, eye_neg])
        bounds_soft = bounds + [(0, None)] * n_constraints
        
        res_soft = linprog(c_soft, A_ub=A_soft, b_ub=b_ub, bounds=bounds_soft, method='highs')
        
        if res_soft.success:
            x_sol = res_soft.x[:n_vars]
            original_obj = np.dot(c, x_sol)
            val = -original_obj if c[0] < 0 else original_obj
            return x_sol[:self.num_vpp_total], val
        else:
            return np.zeros(self.num_vpp_total), 0.0

    def solve_dispatch(self, p_inj_max_profile, p_abs_max_profile, q_ratio=0.5):
        if not self.matrices_ready: return None

        T = p_inj_max_profile.shape[0]
        
        results = {
            'net_inj_max': np.zeros(T), 
            'net_abs_max': np.zeros(T),
            'contrib_inj': {k: np.zeros(T) for k in self.vpp_config},
            'contrib_abs': {k: np.zeros(T) for k in self.vpp_config},
            'phy_inj_sum': np.sum(p_inj_max_profile, axis=1),
            'phy_abs_sum': np.sum(p_abs_max_profile, axis=1)
        }

        # 基础向量
        c_inj = np.hstack([np.ones(self.num_vpp_total), np.zeros(self.num_vpp_total)]) 
        c_abs = np.hstack([-np.ones(self.num_vpp_total), np.zeros(self.num_vpp_total)])
        
        A_P = -self.A_total[:, :self.num_vpp_total]
        A_Q = self.A_total[:, self.num_vpp_total:]
        A_ub = np.hstack([A_P, A_Q])
        b_ub = self.b_total

        print(f"--- 正在计算 {T} 个时段的聚合详情 (已启用 NaN 自动修复) ---")
        
        for t in range(T):
            # 1. 物理边界
            p_max_inj_val = p_inj_max_profile[t, :]
            p_max_abs_val = p_abs_max_profile[t, :]
            q_lim = np.minimum(p_max_inj_val * q_ratio, 300) 
            
            bounds_P = list(zip(-p_max_inj_val, p_max_abs_val))
            bounds_Q = list(zip(-q_lim, q_lim))
            bounds = bounds_P + bounds_Q
            
            # --- A. 计算最大注入 (Max Injection) ---
            p_load_sol, obj_val = self._solve_robust(c_inj, A_ub, b_ub, bounds)
            
            results['net_inj_max'][t] = obj_val
            p_inj_opt = -p_load_sol 
            
            for cat, nodes in self.vpp_config.items():
                idxs = [self.node_to_idx[n] for n in nodes]
                results['contrib_inj'][cat][t] = np.sum(p_inj_opt[idxs])

            # --- B. 计算最大吸收 (Max Absorption) ---
            p_load_sol_abs, obj_val_abs = self._solve_robust(c_abs, A_ub, b_ub, bounds)
            
            results['net_abs_max'][t] = obj_val_abs
            p_abs_opt = p_load_sol_abs 
            
            for cat, nodes in self.vpp_config.items():
                idxs = [self.node_to_idx[n] for n in nodes]
                results['contrib_abs'][cat][t] = np.sum(p_abs_opt[idxs])

        return results

    def plot_stacked_charts(self, results, T=24):
        """绘制堆叠图"""
        hours = np.arange(0, T)
        categories = list(self.vpp_config.keys())
        colors = ['#FFD700', '#87CEEB', '#32CD32', '#FF6347']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Injection
        stack_inj = [results['contrib_inj'][cat] for cat in categories]
        ax1.stackplot(hours, stack_inj, labels=categories, colors=colors, alpha=0.85)
        ax1.plot(hours, results['net_inj_max'], 'k--', linewidth=2, label='Safe Limit (net_inj_max)')
        ax1.plot(hours, results['phy_inj_sum'], 'r:', linewidth=1.5, alpha=0.5, label='Physical Limit (phy_inj_sum)')
        ax1.set_title('Max Injection Capacity', fontweight='bold')
        ax1.set_ylabel('Power (kW)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.4)
        
        # Absorption (取负画图)
        stack_abs = [results['contrib_abs'][cat] for cat in categories]
        stack_abs_neg = [-d for d in stack_abs]
        ax2.stackplot(hours, stack_abs_neg, labels=categories, colors=colors, alpha=0.85)
        ax2.plot(hours, -results['net_abs_max'], 'k--', linewidth=2, label='Safe Limit (net_abs_max)')
        ax2.plot(hours, -results['phy_abs_sum'], 'b:', linewidth=1.5, alpha=0.5, label='Physical Limit (phy_abs_sum)')
        ax2.set_title('Max Absorption Capacity', fontweight='bold')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Power (kW)')
        ax2.grid(True, alpha=0.4)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 3. 真实物理数据生成器 (5倍放大以制造阻塞)
# ==========================================
def generate_realistic_profiles(T=24):
    t = np.arange(T)
    p_inj = np.zeros((T, 5))
    p_abs = np.zeros((T, 5)) 

    # Solar (*5 scale)
    peak, width = 12, 3.0
    solar = np.exp(-((t - peak)**2) / (2 * width**2)) * 3000 * 5
    solar = np.clip(solar + np.random.normal(0, 50*5, T), 0, None)
    p_inj[:, 0], p_inj[:, 1] = solar, solar * 0.5
    p_abs[:, 0:2] = 50 

    # Wind (*5 scale)
    wind = (0.4 + 0.3 * np.sin((t + 3)/24*6.28)) * 3500 * 5
    wind += np.random.uniform(-300, 300, T)
    wind = np.clip(wind, 200, 4000*5)
    p_inj[:, 2], p_abs[:, 2] = wind, 100

    # ESS (*5 scale)
    p_inj[:, 3], p_abs[:, 3] = 1500*5, 1500*5

    # Gas (*5 scale)
    p_inj[:, 4], p_abs[:, 4] = 2000*5, 0

    return p_inj, p_abs

# ==========================================
# 4. 打印报表 (修改列名)
# ==========================================
def print_detailed_report(results):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # Summary Table - 列名已修改
    df_summary = pd.DataFrame({
        'net_inj_max': results['net_inj_max'],
        'phy_inj_sum': results['phy_inj_sum'],
        'net_abs_max': results['net_abs_max'],
        'phy_abs_sum': results['phy_abs_sum']
    })
    # 更新计算列
    df_summary['Curtailment_Inj'] = df_summary['phy_inj_sum'] - df_summary['net_inj_max']
    df_summary.index.name = 'Hour'
    
    print("\n" + "="*80)
    print("【表1】 VPP 聚合总量报表 (列名已更新)")
    print("="*80)
    print(df_summary)

    # Injection Breakdown
    df_contrib_inj = pd.DataFrame(results['contrib_inj'])
    df_contrib_inj['>> TOTAL'] = df_contrib_inj.sum(axis=1)
    df_contrib_inj['TARGET'] = results['net_inj_max']
    df_contrib_inj.index.name = 'Hour'

    print("\n" + "="*80)
    print("【表2】 注入(发电)方向 - 各类资源贡献明细")
    print("="*80)
    print(df_contrib_inj)

    # Absorption Breakdown
    df_contrib_abs = pd.DataFrame(results['contrib_abs'])
    df_contrib_abs['>> TOTAL'] = df_contrib_abs.sum(axis=1)
    df_contrib_abs['TARGET'] = results['net_abs_max']
    df_contrib_abs.index.name = 'Hour'

    print("\n" + "="*80)
    print("【表3】 吸收(用电)方向 - 各类资源贡献明细")
    print("="*80)
    print(df_contrib_abs)

# ==========================================
# 5. 主入口
# ==========================================
if __name__ == "__main__":
    config = {
        'Type A (Solar)': [10, 15],
        'Type B (Wind)':  [18],
        'Type C (ESS)':   [20],
        'Type D (Gas)':   [25]
    }
    
    agg = VPPAggregator(config)
    print("生成仿真数据 (含5倍放大测试)...")
    p_inj_in, p_abs_in = generate_realistic_profiles()
    
    if agg.matrices_ready:
        res = agg.solve_dispatch(p_inj_in, p_abs_in)
        if res:
            print_detailed_report(res)
            print("\n正在绘制堆叠图...")
            agg.plot_stacked_charts(res)