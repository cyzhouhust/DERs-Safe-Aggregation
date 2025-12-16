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

- **app.py**  
  Flask REST API 服务，封装所有算法为 REST 接口

- **requirements.txt**  
  Python 依赖包列表

- **api_examples.md**  
  API 使用示例和文档

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Flask API 服务

```bash
python app.py
```

服务将在 `http://localhost:5123` 启动

### 3. 使用 API

详细的使用示例请参考 [api_examples.md](api_examples.md)

**快速测试：**
```bash
# 健康检查
curl http://localhost:5123/xxaqy-api/health

# 单时间步计算
curl -X POST http://localhost:5123/xxaqy-api/aggregator/solve/single \
  -H "Content-Type: application/json" \
  -d '{
    "vpp_nodes": [10, 15, 18, 20, 25],
    "p_inj_max": [40000, 10000, 40000, 4000, 4000],
    "p_abs_max": [1000, 1000, 1000, 1000, 1000]
  }'
```

---

## API 接口列表

### 核心接口

1. **GET** `/xxaqy-api/health` - 健康检查
2. **POST** `/xxaqy-api/matrices/build` - 构建约束矩阵
3. **POST** `/xxaqy-api/aggregator/init` - 初始化聚合器
4. **POST** `/xxaqy-api/aggregator/solve` - 执行聚合优化计算（多时间步）
5. **POST** `/xxaqy-api/aggregator/solve/single` - 计算单个时间步的聚合能力
6. **POST** `/xxaqy-api/aggregator/clear_cache` - 清除缓存

详细接口文档请参考 [api_examples.md](api_examples.md)

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
