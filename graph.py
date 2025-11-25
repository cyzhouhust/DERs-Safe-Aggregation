import numpy as np
from data import *
# 假设你已经有了 bus_data 和 branch_data
# 之前的预处理代码... (S_base, Z_base 等)

# --- 1. 建立图的拓扑关系 ---
# 节点数 (注意：你的矩阵大小是32x32，因为通常不包含平衡节点Bus 0，或者是指32条支路)
# 这里我们假设我们要计算的是除Slack Bus以外的32个节点的电压
num_buses = 33
num_branches = 32

# 构建父节点映射和支路阻抗映射
# parent_map[节点] = (父节点, 支路索引)
tree_map = {} 
branch_r_pu = branch_data[:, 2] / ((12.66**2)/10) # 确保转换为标么值
branch_x_pu = branch_data[:, 3] / ((12.66**2)/10)

# branch_data 的前两列是 fbus, tbus (假设是 1-based)
for idx, row in enumerate(branch_data):
    f, t = int(row[0])-1, int(row[1])-1 # 转为 0-based
    # 在辐射网中，t 必定是子节点，f 是父节点（假设数据方向正确）
    tree_map[t] = {'parent': f, 'idx': idx, 'r': branch_r_pu[idx], 'x': branch_x_pu[idx]}

# --- 2. 获取每个节点到根节点的路径 ---
def get_path_to_root(node_idx, mapping):
    path = [] # 存储支路索引
    curr = node_idx
    while curr != 0: # 假设0是根节点
        if curr not in mapping:
            break
        info = mapping[curr]
        path.append(info)
        curr = info['parent']
    return path[::-1] # 反转，从根到叶

# 预先计算所有节点的路径
all_paths = {}
# 我们关注的是索引 1 到 32 (即 Bus 2 到 Bus 33)
target_nodes = range(1, num_buses) 

for i in target_nodes:
    all_paths[i] = get_path_to_root(i, tree_map)

# --- 3. 自动构建 R_hat 和 X_hat (电压灵敏度) ---
# 大小 32x32，对应 Bus 2-33
R_hat = np.zeros((32, 32))
X_hat = np.zeros((32, 32))

# 映射关系：矩阵索引 0 对应 Bus 1 (即原始编号2)
# 矩阵索引 i 对应 Bus i+1
for i in range(32):
    bus_i = i + 1
    path_i = all_paths[bus_i]
    for j in range(32):
        bus_j = j + 1
        path_j = all_paths[bus_j]
        
        # 寻找公共路径
        # 比较两个路径列表，累加公共部分的阻抗
        common_r = 0.0
        common_x = 0.0
        
        # 路径是从根出发的，所以可以直接 zip 对比
        for seg_i, seg_j in zip(path_i, path_j):
            if seg_i['idx'] == seg_j['idx']:
                common_r += seg_i['r']
                common_x += seg_i['x']
            else:
                break # 一旦分叉，后续就不再公共
        
        R_hat[i, j] = common_r
        X_hat[i, j] = common_x

print("R_hat 构建完成 (自动计算).")

# --- 4. 自动构建 R_I 和 X_I (电流灵敏度) ---
# 这里的 R_I, X_I 物理意义通常是：
# 支路 k 的功率流 = sum(注入节点 j 的功率)，如果 j 在 k 的下游
# 但你的代码中 R_I 似乎是用来算电压差进而算电流的？
# 根据你的代码: lhs_I = (R_I*P + X_I*Q) / V
# 这意味着 R_I[k, j] = r_k (如果支路k在节点j的路径上)
# 实际上这应该是 "Branch Sensitivity Matrix"

R_I = np.zeros((32, 32))
X_I = np.zeros((32, 32))

# 行 i 代表第 i 条支路 (branch_data的第i行)
# 列 j 代表第 j 个节点 (Bus j+1)
for branch_idx in range(32):
    # 找到这条支路直接连接的末端节点
    # 在 IEEE33 标准数据中，第 k 行支路通常其末端就是节点 k+1 (因为是辐射状且有序)
    # 但为了严谨，我们查找哪个节点的父节点支路是 branch_idx
    child_node = -1
    for node, info in tree_map.items():
        if info['idx'] == branch_idx:
            child_node = node
            break
            
    if child_node == -1: continue

    # 支路 i 的电流受其下游所有节点注入功率的影响
    # 检查哪些节点 j 在 child_node 的下游（或者是 child_node 本身）
    # 方法：检查 branch_idx 是否出现在节点 j 的路径中
    
    current_r = branch_r_pu[branch_idx]
    current_x = branch_x_pu[branch_idx]
    
    for j in range(32):
        bus_j = j + 1
        path_j = all_paths[bus_j]
        
        # 如果 branch_idx 在 bus_j 的通路上
        if any(seg['idx'] == branch_idx for seg in path_j):
            R_I[branch_idx, j] = current_r
            X_I[branch_idx, j] = current_x

print("R_I 构建完成 (自动计算).")

R,X=R_hat,X_hat #单位配平
#R_i, X_I 是电流灵敏度矩阵
R_I,X_I=R_I,X_I
Pd=bus_Pd.reshape(-1,1)#将负荷数据转为列向量
Qd=bus_Qd.reshape(-1,1)
#计算每个节点的电压
V_base = 12.66 # kV
# 计算电压降标么值
V_drop_pu = np.matmul(R, Pd) + np.matmul(X, Qd)
# 计算实际电压 (kV)
VB = V_base * (1.0 - V_drop_pu)
print(VB)
U_max=np.ones(32).reshape(-1,1)*12.66*1.05#电压上限
U_min=np.ones(32).reshape(-1,1)*12.66*0.95#电压下限
U_0=np.ones(32).reshape(-1,1)*12.66
#b=np.vstack((d0,d1))
#print(np.matmul(R,Pd))
print(VB/12.66)
UM=(U_max-U_0)*U_0
um=(U_min-U_0)*U_0

I_M=0.6#线路热限值，单位kA，项目中可以根据实际情况调整

IB=np.zeros(32)
V=np.vstack(([12.66],VB))#横向拼接，得到所有节点电压
z=np.zeros(32)#支路阻抗
for i in range(32):
    fbus = branch_data[i, 0]
    tbus = branch_data[i, 1]
    z_abs = np.sqrt(branch_r[i]**2 + branch_x[i]**2)
    z[i] =z_abs
    IB[i] = (V[int(fbus)-1] - V[int(tbus)-1]) / z_abs
z=z.reshape(-1,1)
print(IB)
#构建电压安全约束参数
gamma_MV,gamma_mV=R/UM,R/um
kai_MV,kai_mV=X/UM,X/um
#构建了电流安全约束参数
gamma_MI=R_I/(I_M*12.66*z)
kai_MI=X_I/(I_M*12.66*z)
#检查一下现在的电压是否越限
lhs_M = np.matmul(gamma_MV, Pd) + np.matmul(kai_MV, Qd)  # 上限
lhs_m = np.matmul(gamma_mV, Pd) + np.matmul(kai_mV, Qd)  # 下限
print(lhs_M),print(lhs_m)
#判断电流是否越限
lhs_I = (np.matmul(R_I, Pd) + np.matmul(X_I, Qd))/(12.66*z)
print(lhs_I)