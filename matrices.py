import numpy as np
from data import bus_Pd, bus_Qd, branch_r, branch_x,branch_data
from itertools import chain


def build_matrices(verbose: bool = False):
    """构建并返回主要的灵敏度矩阵与线性约束项。

    返回顺序: A_V, b_V, A_I, b_I, R_hat, X_hat, R_I, X_I, Pd, Qd
    """
    R_hat, X_hat = np.zeros((32, 32)), np.zeros((32, 32))
    # 主线
    for i in range(17):
        R_hat[i][i], X_hat[i][i] = sum(branch_r[:i + 1]), sum(branch_x[:i + 1])
        # 主线
        for j in range(i, 17):
            R_hat[i][j], X_hat[i][j] = sum(branch_r[:i + 1]), sum(branch_x[:i + 1])
        # 支线1
        for j in range(17, 21):
            R_hat[i][j], X_hat[i][j] = branch_r[0], branch_x[0]
        for j in range(21, 24):
            if i == 0:
                R_hat[i][j], X_hat[i][j] = branch_r[0], branch_x[0]
            else:
                R_hat[i][j], X_hat[i][j] = branch_r[0] + branch_r[1], branch_x[0] + branch_x[1]
        for j in range(24, 32):
            if i == 0:
                R_hat[i][j], X_hat[i][j] = branch_r[0], branch_x[0]
            elif i == 1:
                R_hat[i][j], X_hat[i][j] = branch_r[0] + branch_r[1], branch_x[0] + branch_x[1]
            elif i >= 2 and i <= 4:
                R_hat[i][j], X_hat[i][j] = sum(branch_r[:i + 1]), sum(branch_x[:i + 1])
            else:
                R_hat[i][j], X_hat[i][j] = sum(branch_r[:5]), sum(branch_x[:5])
    # 支线1
    for i in range(17, 21):
        for j in range(i, 21):
            R_hat[i][j], X_hat[i][j] = branch_r[0] + sum(branch_r[17:i + 1]), branch_x[0] + sum(branch_x[17:i + 1])
        for j in range(21, 32):
            R_hat[i][j], X_hat[i][j] = branch_r[0], branch_x[0]
    # 支线2
    for i in range(21, 24):
        for j in range(i, 24):
            R_hat[i][j], X_hat[i][j] = branch_r[0] + sum(branch_r[21:i + 1]), branch_x[0] + sum(branch_x[21:i + 1])
        for j in range(24, 32):
            R_hat[i][j], X_hat[i][j] = branch_r[0] + branch_r[1], branch_x[0] + branch_x[1]
    # 支线3
    for i in range(24, 32):
        for j in range(i, 32):
            R_hat[i][j], X_hat[i][j] = branch_r[0] + sum(branch_r[24:i + 1]), branch_x[0] + sum(branch_x[24:i + 1])
    # 转置
    for i in range(0, 32):
        for j in range(i, 32):
            R_hat[j][i], X_hat[j][i] = R_hat[i][j], X_hat[i][j]

    # 构建电流等效矩阵
    R_I, X_I = np.zeros((32, 32)), np.zeros((32, 32))
    for j in range(0, 32):
        R_I[0][j], X_I[0][j] = branch_r[0], branch_x[0]
    for j in chain(range(1, 16), range(21, 24), range(24, 32)):
        R_I[1][j], X_I[1][j] = branch_r[1], branch_x[1]
    for j in chain(range(2, 16), range(24, 32)):
        R_I[2][j], X_I[2][j] = branch_r[2], branch_x[2]
    for j in chain(range(3, 17), range(24, 32)):
        R_I[3][j], X_I[3][j] = branch_r[3], branch_x[3]
    for i in range(4, 16):
        for j in range(i, 17):
            R_I[i][j], X_I[i][j] = branch_r[i], branch_x[i]
    # 支线1：节点 17~20
    for i in range(17, 20):
        for j in range(i, 21):
            R_I[i][j], X_I[i][j] = branch_r[i], branch_x[i]
    # 支线2：节点 21~23
    for i in range(21, 23):
        for j in range(i, 24):
            R_I[i][j], X_I[i][j] = branch_r[i], branch_x[i]
    # 支线3：节点 24~31
    for i in range(24, 31):
        for j in range(i, 32):
            R_I[i][j], X_I[i][j] = branch_r[i], branch_x[i]

    # R,X是电压灵敏度矩阵（单位换算）
    R, X = R_hat / 1000, X_hat / 1000
    # R_I, X_I 是电流灵敏度矩阵（单位换算）
    R_I, X_I = R_I / 1000, X_I / 1000

    Pd = bus_Pd.reshape(-1, 1)  # 将负荷数据转为列向量
    Qd = bus_Qd.reshape(-1, 1)

    VB = np.ones(32).reshape(-1, 1) * 12.66 - (np.matmul(R, Pd) + np.matmul(X, Qd)) / 12.66
    
    V_max=1.05
    V_min=0.95
    U_max = np.ones(32).reshape(-1, 1) * 12.66 * V_max
    U_min = np.ones(32).reshape(-1, 1) * 12.66 * V_min
    U_0 = np.ones(32).reshape(-1, 1) * 12.66

    UM = (U_max - U_0) * U_0
    um = (U_min - U_0) * U_0
    I_M = 100

    IB = np.zeros(32) 
    V = np.vstack(([12.66], VB))
    
    z = np.zeros(32)
    for i in range(32):
        fbus = branch_data[i, 0]
        tbus = branch_data[i, 1]
        z_abs = np.sqrt(branch_r[i] ** 2 + branch_x[i] ** 2)
        z[i] = z_abs
        IB[i] = (V[int(fbus) - 1] - V[int(tbus) - 1]) / z_abs
    z = z.reshape(-1, 1)
    
    # 构造电压运行安全域参数
    gamma_MV, gamma_mV = R / UM, R / um
    kai_MV, kai_mV = X / UM, X / um
    # 构造电流运行安全域参数
    gamma_MI = R_I / (I_M * 12.66 * z)
    kai_MI = X_I / (I_M * 12.66 * z)
    
    # 电压约束常数项
    # 根据推导: 对于上/下限均可写成
    #   gamma_* * P_vpp + kai_* * Q_vpp <= 1 + gamma_* * Pd + kai_* * Qd
    C_MV_load = np.matmul(gamma_MV, -Pd) + np.matmul(kai_MV, -Qd)
    C_mV_load = np.matmul(gamma_mV, -Pd) + np.matmul(kai_mV, -Qd)
    print(C_MV_load)
    print(C_mV_load)
    # VPP 节点集（假设）
    vpp_nodes = [10, 15, 18, 20, 25]
    #vpp_indices = np.array(vpp_nodes) - 1
    vpp_indices = np.arange(32) # 转为 0-based 索引

    # 提取 VPP 节点的系数矩阵 (32 x |VPP|)，这里是怎么取的
    A_P_MV = gamma_MV[:, vpp_indices]
    A_Q_MV = kai_MV[:, vpp_indices]
    A_P_mV = gamma_mV[:, vpp_indices]
    A_Q_mV = kai_mV[:, vpp_indices]
    
    b_V_ub = 1 - C_MV_load
    b_V_lb = 1 - C_mV_load
    b_V = np.vstack([b_V_ub, b_V_lb])

    A_V_ub = np.hstack([A_P_MV, A_Q_MV])#电压约束矩阵上限部分
    A_V_lb = np.hstack([A_P_mV, A_Q_mV])#电压约束矩阵下限部分
    A_V = np.vstack([A_V_ub, A_V_lb])#电压约束矩阵，纵向拼接上限和下限部分
    
    # 电流约束常数项
    C_MI_load_pos = np.matmul(gamma_MI, -Pd) + np.matmul(kai_MI, -Qd)
    C_MI_load_neg = np.matmul(-gamma_MI, -Pd) + np.matmul(-kai_MI, -Qd)
    print(C_MI_load_pos)
    print(C_MI_load_neg)
    # 提取 VPP 节点的系数矩阵 (32 x |VPP|)
    A_P_MI_pos = gamma_MI[:, vpp_indices]
    A_Q_MI_pos = kai_MI[:, vpp_indices]
    A_P_MI_neg = -gamma_MI[:, vpp_indices]
    A_Q_MI_neg = -kai_MI[:, vpp_indices]
    # 拼接电流约束矩阵
    A_I_pos = np.hstack([A_P_MI_pos, A_Q_MI_pos])
    A_I_neg = np.hstack([A_P_MI_neg, A_Q_MI_neg])
    A_I = np.vstack([A_I_pos, A_I_neg])
    #
    b_I_pos = 1 - C_MI_load_pos
    b_I_neg = 1 - C_MI_load_neg
    b_I = np.vstack([b_I_pos, b_I_neg])

    if verbose:
        print("--- 构建电压约束矩阵示例 ---")
        print(f"VPP 节点索引 (0-based): {vpp_indices}")
        print(f"电压约束矩阵 A_V 的形状: {A_V.shape}")
        print(f"电压约束向量 b_V 的形状: {b_V.shape}")
        print("\n--- 构建电流约束矩阵示例 ---")
        print(f"电流约束矩阵 A_I 的形状: {A_I.shape}")
        print(f"电流约束向量 b_I 的形状: {b_I.shape}")
    return A_V, b_V, A_I, b_I, R_hat, X_hat, R_I, X_I, Pd, Qd, C_MI_load_neg, C_MI_load_pos, C_mV_load, C_MV_load

if __name__ == "__main__":
    A_V, b_V, A_I, b_I, R_hat, X_hat, R_I, X_I, Pd, Qd, C_MI_load_neg, C_MI_load_pos, C_mV_load, C_MV_load = build_matrices(verbose=True)
    print('\nC_mV_load:\n', C_mV_load.flatten())
    print('\nC_MV_load:\n', C_MV_load.flatten())
    print('\nC_MI_load_neg:\n', C_MI_load_neg.flatten())
    print('\nC_MI_load_pos:\n', C_MI_load_pos.flatten())
