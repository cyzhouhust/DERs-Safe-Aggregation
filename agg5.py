import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ==========================================
# 1. 基础配置与模拟矩阵
# ==========================================
GLOBAL_NODES = list(range(2, 34))
GLOBAL_NODE_MAP = {f"bus {n}": i for i, n in enumerate(GLOBAL_NODES)}
FILE_PATH = 'vpp_data.txt'

def build_matrices(verbose=False):
    """模拟电网矩阵 (32节点)"""
    n = 32
    # 模拟简单的电网约束：灵敏度矩阵
    return (np.eye(n)*0.1, np.ones(n)*100, np.eye(n)*0.05, np.ones(n)*100)

def create_vpp_data_file():
    """生成模拟数据文件"""
    headers = "id,type,site,rated_capacity,up_capacity,down_capacity,resp_time,status"
    data_lines = [
        "AC-Cluster-01,空调,bus 10,1.5 MW,600,900,30 min,可用",
        "EV-Hub-03,充电桩,bus 15,0.8 MW,400,300,5 min,约束",
        "PV-Block-12,光伏,bus 18,3.5 MW,3500,0,即时,可用",
        "BESS-02,储能,bus 20,1.0/2.0 MWh,800,800,1s,可用",
        "AC-Cluster-02,空调,bus 25,2.2 MW,880,1320,30 min,可用",
        "Small-Storage,储能,bus 20,0.5MW,200,150,5 min,可用",
        "Wind-Farm-01,风电,bus 30,2.0 MW,2000,0,即时,可用"
    ]
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(headers + "\n")
        f.write("\n".join(data_lines))

# ==========================================
# 2. 数据处理辅助函数
# ==========================================
HEADERS = ['id', 'type', 'site', 'rated_capacity', 'up_capacity', 'down_capacity', 'resp_time', 'status']

def read_and_filter_vpp_data(target_nodes=None):
    if not os.path.exists(FILE_PATH): create_vpp_data_file()
    data = []
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 2: return []
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) == len(HEADERS):
                item = {HEADERS[i]: parts[i].strip() for i in range(len(HEADERS))}
                if '可用' not in item['status']: continue
                if target_nodes and item['site'] not in target_nodes: continue
                data.append(item)
    return data

def get_active_nodes(resources):
    active_nodes_str = sorted(list(set(r['site'] for r in resources)), 
                              key=lambda x: int(x.replace('bus', '').strip()))
    local_map = {name: i for i, name in enumerate(active_nodes_str)}
    return active_nodes_str, local_map

def generate_config_by_type(resources, active_nodes):
    config = {}
    for res in resources:
        r_type = res['type']
        site = res['site']
        if site not in active_nodes: continue
        if r_type not in config: config[r_type] = []
        if site not in config[r_type]: config[r_type].append(site)
    return config

def get_snapshot_bounds(resources, active_nodes, local_map, time_snapshot=12):
    """
    【修改点】只计算单时间点的物理边界
    time_snapshot: 默认取 12:00 的状态
    """
    num_active = len(active_nodes)
    p_inj_bounds = np.zeros(num_active) 
    p_abs_bounds = np.zeros(num_active)
    
    # 模拟时刻因子
    t = time_snapshot
    factor_solar = np.exp(-((t - 12)**2) / (2 * 3.0**2)) # 中午大
    factor_wind = 0.6 + 0.3 * np.sin((t)/24*6.28)        # 随机波动
    factor_flat = 1.0

    for res in resources:
        site = res['site']
        if site not in local_map: continue
        idx = local_map[site]
        
        try:
            up = float(res['up_capacity'])
            down = float(res['down_capacity'])
        except: continue

        r_type = res['type']
        if '光伏' in r_type: profile = factor_solar
        elif '风' in r_type: profile = factor_wind
        else: profile = factor_flat
            
        p_inj_bounds[idx] += up * profile
        p_abs_bounds[idx] += down * profile
        
    return p_inj_bounds, p_abs_bounds

# ==========================================
# 3. 核心聚合器类 (Single Time Step)
# ==========================================
class VPPAggregator1:
    def __init__(self, active_nodes_str, local_map, vpp_config):
        self.active_nodes_str = active_nodes_str 
        self.local_map = local_map              
        self.vpp_config = vpp_config            
        self.num_vars = len(active_nodes_str)    
        
        self.matrices_ready = False
        self._load_and_reduce_matrices()

    def _load_and_reduce_matrices(self):
        ret = build_matrices()
        if ret is None:
            self.matrices_ready = False
            return

        A_V, b_V, A_I, b_I = ret[0], ret[1], ret[2], ret[3]

        # 合并约束矩阵（行方向堆叠）和边界向量
        try:
            A_full = np.vstack([A_V, A_I])
        except Exception:
            # 如果 A_V 或 A_I 不是二维数组，尝试强制转为二维
            A_V2 = np.atleast_2d(A_V)
            A_I2 = np.atleast_2d(A_I)
            A_full = np.vstack([A_V2, A_I2])

        # 边界向量按行拼接为单列向量
        try:
            self.b_total = np.concatenate([np.ravel(b_V), np.ravel(b_I)])
        except Exception:
            self.b_total = np.ravel(b_V).tolist() + np.ravel(b_I).tolist()

        # 确保列数足够（预期为 2 * full_grid_len），不足时用 0 列填充
        full_grid_len = len(GLOBAL_NODES)
        expected_cols = 2 * full_grid_len
        cols = A_full.shape[1]
        if cols < expected_cols:
            pad_cols = expected_cols - cols
            pad = np.zeros((A_full.shape[0], pad_cols))
            A_full = np.hstack([A_full, pad])

        # 计算要提取的列索引（P 和 Q 列）
        target_p_indices = []
        target_q_indices = []
        for bus_name in self.active_nodes_str:
            if bus_name in GLOBAL_NODE_MAP:
                g_idx = GLOBAL_NODE_MAP[bus_name]
                target_p_indices.append(g_idx)
                target_q_indices.append(g_idx + full_grid_len)

        # 提取子矩阵，确保索引有效
        try:
            cols_to_take = target_p_indices + target_q_indices
            self.A_total = A_full[:, cols_to_take]
            self.A_total = np.nan_to_num(self.A_total, nan=0.0)
            self.b_total = np.nan_to_num(self.b_total, nan=0.0)
            self.matrices_ready = True
        except Exception:
            self.matrices_ready = False

    def _solve_robust(self, c, A_ub, b_ub, bounds):
        """通用求解器封装"""
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # linprog 求的是 min c*x，如果是求最大值，c取负，结果取负
            val = -res.fun if c[0] < 0 else res.fun
            return res.x[:self.num_vars], val
        return np.zeros(self.num_vars), 0.0

    def solve_snapshot(self, p_inj_bounds, p_abs_bounds, q_ratio=0.5):
        """
        【核心修改】计算单时间点的聚合能力
        返回: 包含 count, th_val, act_val, ratio 的字典结构
        """
        if not self.matrices_ready: return None
        
        # 1. 构造优化参数
        # 注入目标 (Max Injection -> Min -Sum(P))
        c_inj = np.hstack([np.ones(self.num_vars), np.zeros(self.num_vars)]) 
        # 消纳目标 (Max Absorption -> Min -Sum(P_abs))
        c_abs = np.hstack([-np.ones(self.num_vars), np.zeros(self.num_vars)])
        
        # 约束矩阵 A_ub * [P, Q]^T <= b
        # 注意：这里 P 定义为注入为正，但在矩阵中通常 P_load 为正
        # 这里简化处理：假设矩阵兼容 P_inj 的方向符号
        A_ub = np.hstack([-self.A_total[:, :self.num_vars], self.A_total[:, self.num_vars:]])
        
        # 变量边界
        # P范围: [-Max_Inj, Max_Abs]
        bounds = list(zip(-p_inj_bounds, p_abs_bounds)) + \
                 list(zip(-np.minimum(p_inj_bounds*q_ratio, 300), np.minimum(p_inj_bounds*q_ratio, 300)))
        
        # 2. 执行计算
        # 计算最大注入能力
        sol_inj_vec, act_inj_val = self._solve_robust(c_inj, A_ub, self.b_total, bounds)
        act_inj_val = max(0, -act_inj_val if act_inj_val < 0 else 0) # 修正符号确保为正
        p_inj_opt = -sol_inj_vec # 还原为注入正值

        # 计算最大消纳能力
        sol_abs_vec, act_abs_val = self._solve_robust(c_abs, A_ub, self.b_total, bounds)
        act_abs_val = max(0, act_abs_val)
        p_abs_opt = sol_abs_vec
        
        # 3. 结果统计与格式化
        output = {
            "Up": {"Total": {}, "Details": {}},
            "Down": {"Total": {}, "Details": {}}
        }
        
        # --- 填充 Up (注入) 数据 ---
        theo_inj_total = np.sum(p_inj_bounds)
        output["Up"]["Total"] = {
            "count": self.num_vars, # 聚合节点数
            "th_val": float(f"{theo_inj_total:.2f}"),
            "act_val": float(f"{act_inj_val:.2f}")
        }
        
        for r_type, sites in self.vpp_config.items():
            idxs = [self.local_map[s] for s in sites if s in self.local_map]
            if not idxs: continue
            
            # 统计该类型的各项指标
            sub_count = len(idxs) # 这里的count是指涉及了几个物理节点
            sub_th = np.sum(p_inj_bounds[idxs])
            sub_act = np.sum(p_inj_opt[idxs])
            # 占比 = 该类型实际贡献 / 总实际能力
            ratio = (sub_act / act_inj_val * 100) if act_inj_val > 1e-3 else 0.0
            
            output["Up"]["Details"][r_type] = {
                "count": sub_count,
                "th_val": float(f"{sub_th:.2f}"),
                "act_val": float(f"{sub_act:.2f}"),
                "ratio": float(f"{ratio:.2f}")
            }

        # --- 填充 Down (消纳) 数据 ---
        theo_abs_total = np.sum(p_abs_bounds)
        output["Down"]["Total"] = {
            "count": self.num_vars,
            "th_val": float(f"{theo_abs_total:.2f}"),
            "act_val": float(f"{act_abs_val:.2f}")
        }
        
        for r_type, sites in self.vpp_config.items():
            idxs = [self.local_map[s] for s in sites if s in self.local_map]
            if not idxs: continue
            
            sub_count = len(idxs)
            sub_th = np.sum(p_abs_bounds[idxs])
            sub_act = np.sum(p_abs_opt[idxs])
            ratio = (sub_act / act_abs_val * 100) if act_abs_val > 1e-3 else 0.0
            
            output["Down"]["Details"][r_type] = {
                "count": sub_count,
                "th_val": float(f"{sub_th:.2f}"),
                "act_val": float(f"{sub_act:.2f}"),
                "ratio": float(f"{ratio:.2f}")
            }
            
        return output


# --------- 兼容层: 提供 app.py 期望的接口 ---------
# 默认的资源分组占位（可以由调用方覆盖）
RESOURCE_GROUPS = {}

def solve_snapshot_capacity(t_snapshot=12, target_nodes=None, resource_groups=None):
    """
    兼容旧接口：计算单一时间快照的聚合能力并返回与 `app.py` 期望格式兼容的结果。
    参数:
      - t_snapshot: 时间点（小时）
      - target_nodes: 可选，节点列表，例如 ['bus 10', 'bus 18']
      - resource_groups: 可选，自定义分组（目前仅作为占位）
    返回: 与 `VPPAggregator.solve_snapshot` 相同的字典结构，或 None（失败）
    """
    # 1. 读取资源并筛选（使用已有实现）
    resources = read_and_filter_vpp_data(target_nodes)
    if not resources:
        return None

    # 2. 生成激活节点与本地映射
    active_nodes_str, local_map = get_active_nodes(resources)

    # 3. 生成分组配置（基于资源类型）
    vpp_config = generate_config_by_type(resources, active_nodes_str)

    # 4. 计算物理边界
    p_inj_bounds, p_abs_bounds = get_snapshot_bounds(resources, active_nodes_str, local_map, time_snapshot=t_snapshot)

    # 5. 初始化聚合器并计算快照能力
    agg = VPPAggregator1(active_nodes_str, local_map, vpp_config)
    if not agg.matrices_ready:
        return None

    results = agg.solve_snapshot(p_inj_bounds, p_abs_bounds)

    # 6. 填充 RESOURCE_GROUPS（如果为空，则使用生成的分组）
    global RESOURCE_GROUPS
    if not RESOURCE_GROUPS:
        try:
            RESOURCE_GROUPS = {k: [local_map[s] for s in v if s in local_map] for k, v in vpp_config.items()}
        except Exception:
            RESOURCE_GROUPS = {}

    return results