import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# === DER 设置 ===
DERs = [
    {"P_min": 0, "P_max": 4, "Q_min": -2, "Q_max": 2},
    {"P_min": 1, "P_max": 3, "Q_min": -1, "Q_max": 1},
    {"P_min": 0, "P_max": 2, "Q_min": -0.5, "Q_max": 0.5}
]

# Crisp DOSS 参数（公式55）这是本文的核心
gamma = [0.5, 0.8, 0.6]  # γ_j^vpp
chi = [0.4, 0.7, 0.8]    # χ_j^vpp
omega = 6.0              # ϖ

# === 角度定义 ===
phi_list = np.linspace(0, 2 * np.pi, 180)
boundary_doss = []
boundary_no_doss = []

# === 遍历每个方向角 ===
for phi in phi_list:
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)

    num_ders = len(DERs)
    c = -np.array([cos_phi for _ in range(num_ders)])  # 最大化方向投影
    bounds = [(der["P_min"], der["P_max"]) for der in DERs]

    # === 无 DOSS 限制 ===
    res_no_doss = linprog(c, bounds=bounds)
    if res_no_doss.success:
        P_list = res_no_doss.x
        P_vpp = np.sum(P_list)
        Q_vpp = P_vpp * tan_phi if np.isfinite(tan_phi) else 0
        if np.abs(Q_vpp) < 100:  # 防止发散
            boundary_no_doss.append([P_vpp, Q_vpp])

    # === 有 DOSS 限制 ===
    A_doss = [[gamma[j] + chi[j] * tan_phi for j in range(num_ders)]]
    b_doss = [omega]

    res_doss = linprog(c, A_ub=A_doss, b_ub=b_doss, bounds=bounds)
    if res_doss.success:
        P_list = res_doss.x
        P_vpp = np.sum(P_list)
        Q_vpp = P_vpp * tan_phi if np.isfinite(tan_phi) else 0
        if np.abs(Q_vpp) < 100:
            boundary_doss.append([P_vpp, Q_vpp])

# === 可视化 ===
boundary_doss = np.array(boundary_doss)
boundary_no_doss = np.array(boundary_no_doss)

plt.figure(figsize=(7, 7))
plt.plot(boundary_no_doss[:, 0], boundary_no_doss[:, 1], 'g--', label='Without DOSS')
plt.fill(boundary_no_doss[:, 0], boundary_no_doss[:, 1], color='lightgreen', alpha=0.3)

plt.plot(boundary_doss[:, 0], boundary_doss[:, 1], 'b-', label='With Crisp DOSS')
plt.fill(boundary_doss[:, 0], boundary_doss[:, 1], color='lightblue', alpha=0.5)

plt.xlabel(r'$P^{\mathrm{vpp}}$ (kW)')
plt.ylabel(r'$Q^{\mathrm{vpp}}$ (kVar)')
plt.title('Flexibility Region: With vs Without DOSS')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
