import numpy as np
from agg4 import VPPAggregator
# from vpp_aggregator import VPPAggregator # 假设你把上面的类保存到了文件

# --- 1. 配置参数 ---
vpp_nodes = [10, 15, 18, 20, 25]
T = 24
num_vpp = len(vpp_nodes)

# --- 2. 准备输入数据 (模拟资源的上调/下调容量) ---
# 这里对应你原始代码中的 "I. 模拟 24 小时..." 部分
# 我们现在先生成好数据，再传给类
base_p_max = np.array([40000, 10000, 40000, 4000, 4000])
base_p_min_abs = np.array([1000, 1000, 1000, 1000, 1000])

p_inj_max_profile = np.zeros((T, num_vpp))
p_abs_max_profile = np.zeros((T, num_vpp))

for t in range(T):
    hour = t + 1
    scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi)
    
    # 构造每个时刻的容量矩阵 (行: 时间, 列: 节点)
    p_inj_max_profile[t, :] = base_p_max * scale
    # 吸收容量 (输入应为正值，表示容量大小)
    p_abs_max_profile[t, :] = base_p_min_abs * (1 + 0.5 * scale)

# --- 3. 实例化并计算 ---
# 初始化聚合器
aggregator = VPPAggregator(vpp_nodes)

# 执行优化
# 输入：所有时段的所有节点上调容量，所有时段的所有节点下调容量
results = aggregator.solve_dispatch(p_inj_max_profile, p_abs_max_profile)

# --- 4. 输出与绘图 ---
if results:
    print("\n--- 24小时聚合结果 ---")
    print(f"最大安全上调容量 (kW): \n{results['net_inj_max']}")
    print(f"最大安全下调容量 (kW): \n{results['net_abs_max']}")
    
    # 调用类的绘图方法
    aggregator.plot_results(results, T)
