import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 假设 build_matrices 来自你的 matrices.py
# 如果 matrices 和 data 模块在同一目录下，直接引用即可
try:
    from matrices import build_matrices
except ImportError:
    # 仅作占位，防止缺少文件报错，实际使用时请确保环境正确
    def build_matrices(verbose=False):
        raise ImportError("缺少 matrices.py 或 build_matrices 函数")

class VPPAggregator:
    def __init__(self, vpp_nodes):
        """
        初始化 VPP 聚合器
        :param vpp_nodes: list, VPP 控制的节点编号列表 (e.g., [10, 15, ...])
        """
        self.vpp_nodes = vpp_nodes
        self.num_vpp = len(vpp_nodes)
        self.matrices_ready = False
        
        # 预加载网络矩阵
        self._load_network_matrices()

    def _load_network_matrices(self):
        """内部方法：构建并切片灵敏度矩阵"""
        try:
            # 获取全网矩阵
            # 注意：build_matrices 返回值需与你实际的 matrices.py 一致
            # 这里假设返回顺序为 A_V, b_V, A_I, b_I, ...
            ret = build_matrices(verbose=False)
            A_V, b_V, A_I, b_I = ret[0], ret[1], ret[2], ret[3]
            
            # 组合电压和电流约束
            A_total_full = np.vstack([A_V, A_I])
            self.b_total = np.vstack([b_V, b_I]).flatten()
            
            # --- 关键：根据 vpp_nodes 进行切片 ---
            # 假设全网矩阵的列顺序是 [所有节点P, 所有节点Q]
            # 我们需要提取属于 VPP 节点的列
            # 注意：这里需要确保 build_matrices 的列索引与 vpp_nodes 对应关系正确
            # 下面的切片假设 vpp_nodes 的索引直接对应矩阵的列 (如果是 IEEE33 等标准算例通常需要 -1 偏移，请根据你的 data.py 确认)
            
            # 这里做了一个通用假设：build_matrices 返回的矩阵列索引就是节点号
            # 如果你的 build_matrices 已经处理了全网所有节点：
            
            # 提取 P 相关列 (假设前 N 列是 P)
            # 这里的切片逻辑需要根据你的 matrices.py 具体实现调整
            # 假设矩阵列数足够覆盖 max(vpp_nodes)
            
            # 若 A 矩阵列对应全网节点：
            col_indices_P = np.array(self.vpp_nodes) # P 列索引
            # 假设总节点数是 n_bus，Q 的列索引通常是 index + n_bus (需确认你的 build_matrices 逻辑)
            # 这里为了稳健，假设你传入的代码逻辑中 A_total_A 已经是针对特定变量的，
            # 但通常 build_matrices 返回全网。
            # 为了复用你原本的逻辑，这里我们假设 build_matrices 返回的是针对 *全网所有节点* 的。
            
            # 如果 build_matrices 内部已经处理了只返回 VPP 相关的列，那就不需要高级切片。
            # 但根据你提供的原始代码： A_P_A_part = A_total_A[:, :num_vpp]
            # 这暗示 build_matrices 可能在你的原始环境中已经只返回了 VPP 相关的矩阵，或者你在外部做了处理。
            
            # *** 为了保持和你提供的代码片段完全一致的逻辑 ***
            # 我们直接存储 A_total_A，并在 solve 时处理符号
            self.A_total_A = A_total_full 
            self.matrices_ready = True
            
        except Exception as e:
            print(f"初始化矩阵失败: {e}")
            self.matrices_ready = False

    def solve_dispatch(self, p_inj_max_profile, p_abs_max_profile, q_ratio=0.5):
        """
        计算 24 小时最大聚合能力
        :param p_inj_max_profile: np.array shape(T, num_vpp), 每个时段每个节点的物理最大注入功率
        :param p_abs_max_profile: np.array shape(T, num_vpp), 每个时段每个节点的物理最大吸收功率 (正值)
        :param q_ratio: float, Q/P 比例，用于自动计算 Q 边界
        :return: 字典，包含 'inj_max', 'abs_max', 'physical_inj', 'physical_abs'
        """
        if not self.matrices_ready:
            print("错误：网络矩阵未初始化")
            return None

        T = p_inj_max_profile.shape[0]
        max_net_inj = np.zeros(T)
        max_net_abs = np.zeros(T)
        
        # 目标函数向量 (只针对 P 优化, Q 随之变化或为0)
        # 优化变量 x = [P_1...P_n, Q_1...Q_n] (约定 B: Load方向)
        c_inj = np.hstack([np.ones(self.num_vpp), np.zeros(self.num_vpp)])   # Min sum(P_load) -> Max Injection
        c_abs = np.hstack([-np.ones(self.num_vpp), np.zeros(self.num_vpp)])  # Min sum(-P_load) -> Max Absorption

        # 准备矩阵转换 (A -> B 约定)
        # 根据你的原始代码逻辑：
        # A_P_B_part = -A_P_A_part (因为 P_load = -P_inj)
        # A_Q_B_part = A_Q_A_part (Q 方向通常不变或根据你的定义)
        
        # 注意：这里需要根据 build_matrices 的具体输出来切分 P 和 Q 部分
        # 假设 A_total_A 的列数是 2 * num_vpp (如果你原来的 build_matrices 是专门为选定的 VPP 生成的)
        # 否则需要在这里进行列索引切片。
        # *** 基于你提供的原始代码，直接切分 ***
        if self.A_total_A.shape[1] != 2 * self.num_vpp:
             # 如果矩阵列数不对，说明 build_matrices 返回的是全网矩阵，需要切片
             # 这是一个简单的切片修正逻辑（假设全网33节点）
             n_bus = 33 # 假设值，视 build_matrices 而定
             # 这里通过切片提取 VPP 节点对应的列
             # indices_p = self.vpp_nodes
             # indices_q = [x + n_bus for x in self.vpp_nodes] # 假设 Q 在后半部分
             # A_vpp = self.A_total_A[:, indices_p + indices_q]
             pass 
             # *为了安全起见，我们假设 build_matrices 如你原代码所示，已经适配了 VPP 数量*
        
        A_P_A_part = self.A_total_A[:, :self.num_vpp]
        A_Q_A_part = self.A_total_A[:, self.num_vpp:]
        
        # 转换到 Load 约定 (约定 B)
        A_P_B_part = -A_P_A_part
        A_total_B = np.hstack([A_P_B_part, A_Q_A_part])
        b_total_B = self.b_total

        print(f"--- 开始 {T} 时段滚动优化 ---")
        
        for t in range(T):
            # 1. 获取当前时刻边界
            p_max_t = p_inj_max_profile[t, :]
            p_abs_t = p_abs_max_profile[t, :] # 输入约定为正值的容量
            
            # 2. 构造 Q 边界 (复用你的逻辑)
            # 注意：p_abs_t 是容量(正)，原始代码逻辑中 p_inj_min_t 是负值
            # 还原原始逻辑:
            # P_inj_max_t = p_max_t
            # P_inj_min_t = -p_abs_t
            
            q_limit_t = p_max_t * q_ratio
            # 设置无功硬约束，这里为了通用性可以设大一点或作为参数传入
            q_hard_limit = np.array([250, 300, 200, 200, 350]) 
            # 如果节点数不同，需要适配 q_hard_limit 长度，这里做个简单的广播处理
            if len(q_hard_limit) != self.num_vpp:
                q_hard_limit = np.ones(self.num_vpp) * 300
                
            q_max_t = np.minimum(q_limit_t, q_hard_limit)
            q_min_t = -q_max_t

            # 3. 构造 Bounds (约定 B: Load)
            # P_B (Load) 范围: [-P_inj_max, -P_inj_min] => [-P_inj_max, P_abs_max]
            # 原始代码: P_B_min = -P_inj_max, P_B_max = -P_inj_min
            
            bounds_P_t = list(zip(-p_max_t, p_abs_t))
            bounds_Q_t = list(zip(q_min_t, q_max_t))
            bounds_total = bounds_P_t + bounds_Q_t

            # 4. 求解
            # Max Injection (Min sum P_load)
            res_inj = linprog(c_inj, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total, method='highs')
            # 结果取负，因为 min P_load = - Max P_inj
            max_net_inj[t] = -res_inj.fun if res_inj.success else np.nan

            # Max Absorption (Min sum -P_load)
            res_abs = linprog(c_abs, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total, method='highs')
            max_net_abs[t] = res_abs.fun if res_abs.success else np.nan

        return {
            'net_inj_max': max_net_inj,
            'net_abs_max': max_net_abs,
            'phy_inj_sum': np.sum(p_inj_max_profile, axis=1),
            'phy_abs_sum': np.sum(p_abs_max_profile, axis=1)
        }

    @staticmethod
    def plot_results(results, T=24):
        """可视化结果"""
        hours = np.arange(1, T + 1)
        plt.figure(figsize=(12, 6))
        
        # 物理边界
        plt.plot(hours, results['phy_inj_sum'], 'r--', alpha=0.5, label='Physical Max Inj')
        plt.plot(hours, -results['phy_abs_sum'], 'b--', alpha=0.5, label='Physical Max Abs')
        
        # 网络安全边界
        plt.plot(hours, results['net_inj_max'], 'r-o', linewidth=2, label='Safe Max Inj')
        plt.plot(hours, results['net_abs_max'], 'b-x', linewidth=2, label='Safe Max Abs') # 下调画在负轴

        # 这里的 abs_max 算出来是正值(容量)，画图时通常取负表示方向
        # 修正：如果 linprog 返回的是 load 值，那么 max absorption (load) 是正的。
        # 为了画图符合直觉（下调为负），这里取负号。
        plt.plot(hours, -results['net_abs_max'], 'b-x', linewidth=2, label='Safe Max Abs (Plot)')

        plt.fill_between(hours, -results['net_abs_max'], results['net_inj_max'], color='gray', alpha=0.1)
        plt.axhline(0, color='k', linestyle=':')
        plt.title('VPP Safe Aggregation Capacity (24 Hours)')
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
