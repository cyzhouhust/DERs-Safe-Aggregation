## 文件说明

- **data.py**  
  原始数据格式，包括配电网数据、线路拓扑数据和负荷数据。

- **matrices.py**  
  构建电压、电流约束参数。

- **agg3.py**  
  聚合算法执行。

---

### 主要参数

- `vpp_nodes`  
  选择用于聚合的资源节点

- `BASE_P_MAX`  
  资源最大上调容量

- `BASE_P_MIN_ABS`  
  资源最大下调容量

- `max_network_injection`  
  最大安全聚合上行容量

- `max_network_absorption`  
  最大安全聚合下行容量
