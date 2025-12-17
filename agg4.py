import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import linprog

# ==========================================
# 1. 全网参照系 (Global Reference)
# ==========================================
# 这是 IEEE 33 节点系统的物理真理，永远不变。
# 用于告诉程序：原始矩阵的第几列对应哪个节点。
GLOBAL_NODES = list(range(2, 34)) # Node 2 ~ Node 33
GLOBAL_NODE_MAP = {f"bus {n}": i for i, n in enumerate(GLOBAL_NODES)}
FILE_PATH = 'vpp_data.txt'
# 导入矩阵库
try:
    from matrices import build_matrices
except ImportError:
    print("⚠️ 提示: 未找到 matrices.py，使用模拟矩阵。")
    def build_matrices(verbose=False):
        # 模拟一个全网矩阵 (32节点 * 2变量 = 64列)
        n = 32 
        # 这是一个 "胖" 矩阵：64 行约束(假设)，64 列变量
        # 左边 32 列是 P，右边 32 列是 Q
        return (np.eye(n)*0.1, np.ones(n)*100, np.eye(n)*0.05, np.ones(n)*100)
# ==========================================
# 2. 数据文件生成
# ==========================================
def create_vpp_data_file():
    headers = "id,type,site,rated_capacity,up_capacity,down_capacity,resp_time,status"
    data_lines = [
        "AC-Cluster-01,空调,bus 10,1.5 MW,600,900,30 min,可用",
        "EV-Hub-03,充电桩,bus 15,0.8 MW,400,300,5 min,约束",
        "PV-Block-12,光伏,bus 18,3.5 MW,3500,0,即时,可用",
        "BESS-02,储能,bus 20,1.0/2.0 MWh,800,800,1s,可用",
        "AC-Cluster-02,空调,bus 25,2.2 MW,880,1320,30 min,可用",
        "dds,储能,bus 20,0.5MW,200,150,5 min,可用",
        "Wind-Farm-01,风电,bus 30,2.0 MW,2000,0,即时,可用"
    ]
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(headers + "\n")
        f.write("\n".join(data_lines))
    print(f"✅ 已创建数据文件 {FILE_PATH}")
 
# ==========================================
# 3. 资源读取与动态配置
# ==========================================
HEADERS = ['id', 'type', 'site', 'rated_capacity', 'up_capacity', 'down_capacity', 'resp_time', 'status']

def read_and_filter_vpp_data(target_nodes=None):
    if not os.path.exists(FILE_PATH): create_vpp_data_file()
    
    data = []
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(',')
            if len(parts) == len(HEADERS):
                item = {HEADERS[i]: parts[i].strip() for i in range(len(HEADERS))}
                if '可用' not in item['status']: continue
                if target_nodes and item['site'] not in target_nodes: continue
                data.append(item)
    return data

def get_active_nodes(resources):
    """
    从筛选后的资源中，提取出实际上涉及了哪些节点。
    例如：只选了 bus 10 和 bus 18。
    """
    # 使用 set 去重，然后排序确保顺序一致
    active_nodes_str = sorted(list(set(r['site'] for r in resources)), 
                              key=lambda x: int(x.replace('bus', '').strip()))
    
    # 建立一个新的、迷你的映射表
    # 例如: {'bus 10': 0, 'bus 18': 1} -> 只有 2 列
    local_map = {name: i for i, name in enumerate(active_nodes_str)}
    
    return active_nodes_str, local_map

def convert_static_to_profiles(resources, active_nodes, local_map, T=24):
    """
    只生成 "选中节点" 的曲线，矩阵宽度等于选中节点数
    """
    num_active = len(active_nodes)
    p_inj = np.zeros((T, num_active)) 
    p_abs = np.zeros((T, num_active)) 
    t = np.arange(T)
    
    # 曲线模板
    curve_solar = np.exp(-((t - 12)**2) / (2 * 3.0**2))
    curve_wind = np.clip(0.4 + 0.3 * np.sin((t + 3)/24*6.28) + np.random.uniform(-0.1, 0.1, T), 0.1, 1.0)
    curve_flat = np.ones(T)

    print(f"\n--- 正在生成动态曲线 (仅针对 {num_active} 个选中节点) ---")
    
    for res in resources:
        site = res['site']
        # 这里用 local_map，只映射到缩减后的矩阵索引
        if site not in local_map: continue
        idx = local_map[site] 
        
        try:
            up = float(res['up_capacity'])
            down = float(res['down_capacity'])
        except: continue

        r_type = res['type']
        if '光伏' in r_type: profile = curve_solar
        elif '风' in r_type: profile = curve_wind
        else: profile = curve_flat 
            
        p_inj[:, idx] += up * profile
        p_abs[:, idx] += down * profile 
        
    return p_inj, p_abs

def generate_config_by_type(resources, active_nodes):
    """生成分组配置，用于画图"""
    config = {}
    for res in resources:
        r_type = res['type']
        site = res['site']
        if site not in active_nodes: continue
        
        if r_type not in config: config[r_type] = []
        if site not in config[r_type]: config[r_type].append(site)
    return config

# ==========================================
# 4. 优化核心类 (动态提取版)
# ==========================================
class VPPAggregator:
    def __init__(self, active_nodes_str, local_map, vpp_config):
        self.active_nodes_str = active_nodes_str # ['bus 10', 'bus 18']
        self.local_map = local_map               # {'bus 10': 0, 'bus 18': 1}
        self.vpp_config = vpp_config             # 分组信息
        self.num_vars = len(active_nodes_str)    # 变量数 (例如 2)
        
        self.matrices_ready = False
        self._load_and_reduce_matrices()

    def _load_and_reduce_matrices(self):
        """
        核心逻辑：加载全网矩阵 -> 提取特定列 -> 形成微型矩阵
        """
        ret = build_matrices(verbose=False)
        if ret is None: return
        
        # 1. 获取原始全网矩阵
        A_V, b_V, A_I, b_I = ret[0], ret[1], ret[2], ret[3]
        A_full = np.vstack([A_V, A_I])
        self.b_total = np.vstack([b_V, b_I]).flatten()
        
        # 2. 计算需要 "抠" 哪几列
        # 我们需要知道 'bus 10' 在原始全网矩阵里是第几列
        full_grid_len = len(GLOBAL_NODES) # 32
        
        target_p_indices = []
        target_q_indices = []
        
        print(f"DEBUG: 正在从全网矩阵中提取 {self.active_nodes_str} 的参数...")
        
        for bus_name in self.active_nodes_str:
            # 查全局表：bus 10 -> global index 8
            if bus_name in GLOBAL_NODE_MAP:
                g_idx = GLOBAL_NODE_MAP[bus_name]
                
                # P 的列索引就是 global index
                target_p_indices.append(g_idx)
                
                # Q 的列索引通常在后半段 (偏移量 = 32)
                # 注意：这里假设 build_matrices 返回的是标准的 P前Q后 结构
                # 如果 build_matrices 的列数不足，这里要做容错
                if A_full.shape[1] >= 2 * full_grid_len:
                    target_q_indices.append(g_idx + full_grid_len)
                else:
                    # 如果模拟矩阵比较小，做个简单的 fallback
                    target_q_indices.append(g_idx + (A_full.shape[1]//2))
            else:
                print(f"⚠️ 警告: {bus_name} 不在全网参照表中，将被忽略。")

        # 3. 执行提取 (Slicing)
        # self.A_total 的形状将是 (Constraints, 2*num_vars)
        # 例如选了2个节点，这就变成一个只有 4 列的矩阵
        try:
            self.A_total = A_full[:, target_p_indices + target_q_indices]
            print(f"DEBUG: 提取成功！优化矩阵形状: {self.A_total.shape} (行数 x 变量数)")
        except IndexError as e:
            print(f"❌ 提取失败，矩阵索引越界: {e}")
            self.matrices_ready = False
            return

        self.A_total = np.nan_to_num(self.A_total, nan=0.0)
        self.b_total = np.nan_to_num(self.b_total, nan=0.0)
        self.matrices_ready = True

    def _solve_robust(self, c, A_ub, b_ub, bounds):
        # 这里的 num_vars 是动态的，比如 2
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            val = -res.fun if c[0] < 0 else res.fun
            return res.x[:self.num_vars], val
        
        # 软约束逻辑
        n_con = len(b_ub)
        c_soft = np.concatenate([c, np.ones(n_con) * 1e5])
        A_soft = np.hstack([A_ub, -np.eye(n_con)])
        bounds_soft = bounds + [(0, None)] * n_con
        res_soft = linprog(c_soft, A_ub=A_soft, b_ub=b_ub, bounds=bounds_soft, method='highs')
        if res_soft.success:
            x_sol = res_soft.x[:len(c)]
            val = np.dot(c, x_sol)
            return x_sol[:self.num_vars], -val if c[0]<0 else val
        return np.zeros(self.num_vars), 0.0

    def solve_dispatch(self, p_inj, p_abs, q_ratio=0.5):
        if not self.matrices_ready: return None
        T = p_inj.shape[0]
        
        results = {
            'net_inj_max': np.zeros(T), 'net_abs_max': np.zeros(T),
            'contrib_inj': {k: np.zeros(T) for k in self.vpp_config},
            'contrib_abs': {k: np.zeros(T) for k in self.vpp_config},
            'phy_inj_sum': np.sum(p_inj, axis=1),
            'phy_abs_sum': np.sum(p_abs, axis=1)
        }
        
        # 动态构造系数向量 c，长度 = 2 * num_vars
        c_inj = np.hstack([np.ones(self.num_vars), np.zeros(self.num_vars)]) 
        c_abs = np.hstack([-np.ones(self.num_vars), np.zeros(self.num_vars)])
        
        # 构造优化矩阵
        A_ub = np.hstack([-self.A_total[:, :self.num_vars], self.A_total[:, self.num_vars:]])
        
        print(f"--- 优化计算进行中 (变量数: {self.num_vars*2}) ---")
        
        for t in range(T):
            bounds = list(zip(-p_inj[t], p_abs[t])) + \
                     list(zip(-np.minimum(p_inj[t]*q_ratio, 300), np.minimum(p_inj[t]*q_ratio, 300)))
            
            # Injection
            sol_inj, val_inj = self._solve_robust(c_inj, A_ub, self.b_total, bounds)
            results['net_inj_max'][t] = -val_inj if val_inj < 0 else 0 
            
            p_inj_opt = -sol_inj
            for k, site_list in self.vpp_config.items():
                # 使用 local_map 查找缩减后的小索引
                idxs = [self.local_map[site] for site in site_list if site in self.local_map]
                if idxs:
                    results['contrib_inj'][k][t] = np.sum(p_inj_opt[idxs])

            # Absorption
            sol_abs, val_abs = self._solve_robust(c_abs, A_ub, self.b_total, bounds)
            results['net_abs_max'][t] = val_abs
            for k, site_list in self.vpp_config.items():
                idxs = [self.local_map[site] for site in site_list if site in self.local_map]
                if idxs:
                    results['contrib_abs'][k][t] = np.sum(sol_abs[idxs])
                
        return results

    def plot_stacked_charts(self, results, T=24):
        hours = np.arange(T)
        cats = list(self.vpp_config.keys())
        if not cats: return
        cmap = cm.get_cmap('tab10', len(cats))
        colors = [cmap(i) for i in range(len(cats))]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        ax1.stackplot(hours, [results['contrib_inj'][c] for c in cats], labels=cats, colors=colors, alpha=0.85)
        ax1.plot(hours, results['net_inj_max'], 'k--', linewidth=2, label='Safe Limit')
        ax1.plot(hours, results['phy_inj_sum'], 'r:', label='Physical Limit')
        ax1.set_title(f'Max Injection (Active Nodes: {self.num_vars})')
        ax1.legend(loc='upper right')
        
        ax2.stackplot(hours, [-results['contrib_abs'][c] for c in cats], labels=cats, colors=colors, alpha=0.85)
        ax2.plot(hours, -results['net_abs_max'], 'k--', linewidth=2, label='Safe Limit')
        ax2.plot(hours, -results['phy_abs_sum'], 'b:', label='Physical Limit')
        ax2.set_title('Max Absorption')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 5. 主程序
# ==========================================
def print_detailed_report(results):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    df = pd.DataFrame({
        'net_inj_max': results['net_inj_max'],
        'phy_inj_sum': results['phy_inj_sum'],
        'net_abs_max': results['net_abs_max'],
        'phy_abs_sum': results['phy_abs_sum']
    })
    df['Curtailment'] = df['phy_inj_sum'] - df['net_inj_max']
    print("\n【表1】 聚合总量报表")
    print(df.head(12)) 
    print("\n【表2】 各类资源贡献 (前5行)")
    print(pd.DataFrame(results['contrib_inj']).head())

if __name__ == "__main__":
    create_vpp_data_file()
    print(f"\n系统准备就绪。")
    user_input = input("请输入聚合节点(如 bus 10, bus 18)，回车全选: ").strip()
    target_nodes = [n.strip() for n in user_input.split(',')] if user_input else None
    # 1. 筛选资源
    resources = read_and_filter_vpp_data(target_nodes)
    if not resources:
        print("无资源可聚合。")
        sys.exit()
    # 2. 【核心】分析出实际上激活了哪些节点
    # active_nodes: ['bus 10', 'bus 18']
    # local_map: {'bus 10': 0, 'bus 18': 1}
    active_nodes_str, local_map = get_active_nodes(resources)
    print(f"激活节点: {active_nodes_str}")
    # 3. 准备数据
    dynamic_config = generate_config_by_type(resources, active_nodes_str)
    # 这里的矩阵只有 2 列宽 (如果选了2个节点)
    p_inj, p_abs = convert_static_to_profiles(resources, active_nodes_str, local_map)
    # 4. 初始化聚合器 (传入缩减后的映射)
    agg = VPPAggregator(active_nodes_str, local_map, dynamic_config)
    if agg.matrices_ready:
        res = agg.solve_dispatch(p_inj, p_abs)
        if res:
            print_detailed_report(res)
            agg.plot_stacked_charts(res)