## 文件说明

- **data.py**  
  原始数据格式，包括配电网数据、线路拓扑数据和负荷数据。

- **matrices.py**  
  构建电压、电流约束参数。

- **agg3.py**  
  聚合算法执行。

- **agg4.py**  
  聚合代码封装，vppAGGregator,负荷核心逻辑、矩阵构建和优化求解

- **agg_demo.py**  
  展示如何准备数据并调用上面的类，将监测的负荷数据节点位置，上调和下调容量读取，调用这个类进行计算，得到安全汇聚可调范围
---

### 主要参数

- `vpp_nodes`  
  选择用于聚合的资源节点

- `BASE_P_MAX`
- `p_inj_max_profile`   
  资源最大上调容量

- `BASE_P_MIN_ABS`
- `p_abs_max_profile`  
  资源最大下调容量

- `max_network_injection`  
  最大安全聚合上行容量

- `max_network_absorption`  
  最大安全聚合下行容量
