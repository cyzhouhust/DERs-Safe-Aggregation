# API 使用示例

## 启动服务

```bash
python app.py
```

服务将在 `http://localhost:5123` 启动

## API 接口说明

### 1. 健康检查

**GET** `/xxaqy-api/health`

**响应示例:**
```json
{
  "status": "healthy",
  "message": "DERs Safe Aggregation API is running"
}
```

---

### 2. 构建约束矩阵

**POST** `/xxaqy-api/matrices/build`

**请求体:**
```json
{
  "verbose": false,
  "vpp_nodes": [10, 15, 18, 20, 25]
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "A_V": [[...], [...]],
    "b_V": [[...], [...]],
    "A_I": [[...], [...]],
    "b_I": [[...], [...]],
    "R_hat": [[...], [...]],
    "X_hat": [[...], [...]],
    "R_I": [[...], [...]],
    "X_I": [[...], [...]],
    "Pd": [...],
    "Qd": [...],
    "C_MI_load_neg": [...],
    "C_MI_load_pos": [...],
    "C_mV_load": [...],
    "C_MV_load": [...],
    "shapes": {
      "A_V": [64, 10],
      "b_V": [64, 1],
      ...
    }
  },
  "vpp_nodes": [10, 15, 18, 20, 25]
}
```

---

### 3. 初始化聚合器

**POST** `/xxaqy-api/aggregator/init`

**请求体:**
```json
{
  "vpp_nodes": [10, 15, 18, 20, 25]
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "Aggregator initialized successfully",
  "vpp_nodes": [10, 15, 18, 20, 25],
  "cache_key": "10,15,18,20,25"
}
```

---

### 4. 执行聚合优化计算（多时间步）

**POST** `/xxaqy-api/aggregator/solve`

**请求体:**
```json
{
  "vpp_nodes": [10, 15, 18, 20, 25],
  "p_inj_max_profile": [
    [40000, 10000, 40000, 4000, 4000],
    [38000, 9500, 38000, 3800, 3800],
    ...
  ],
  "p_abs_max_profile": [
    [1000, 1000, 1000, 1000, 1000],
    [1100, 1100, 1100, 1100, 1100],
    ...
  ],
  "q_ratio": 0.5,
  "use_cache": true
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "net_inj_max": [85000.5, 82000.3, ...],
    "net_abs_max": [4500.2, 4800.1, ...],
    "phy_inj_sum": [90000.0, 87000.0, ...],
    "phy_abs_sum": [5000.0, 5500.0, ...]
  },
  "metadata": {
    "vpp_nodes": [10, 15, 18, 20, 25],
    "num_time_steps": 24,
    "num_vpp_nodes": 5,
    "q_ratio": 0.5
  }
}
```

---

### 5. 计算单个时间步的聚合能力

**POST** `/xxaqy-api/aggregator/solve/single`

**请求体:**
```json
{
  "vpp_nodes": [10, 15, 18, 20, 25],
  "p_inj_max": [40000, 10000, 40000, 4000, 4000],
  "p_abs_max": [1000, 1000, 1000, 1000, 1000],
  "q_ratio": 0.5
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "net_inj_max": 85000.5,
    "net_abs_max": 4500.2,
    "phy_inj_sum": 90000.0,
    "phy_abs_sum": 5000.0
  },
  "metadata": {
    "vpp_nodes": [10, 15, 18, 20, 25],
    "q_ratio": 0.5
  }
}
```

---

### 6. 清除缓存

**POST** `/xxaqy-api/aggregator/clear_cache`

**响应示例:**
```json
{
  "success": true,
  "message": "Cleared 3 cached aggregator(s)"
}
```

---

## 使用 curl 测试示例

### 健康检查
```bash
curl http://localhost:5123/xxaqy-api/health
```

### 初始化聚合器
```bash
curl -X POST http://localhost:5123/xxaqy-api/aggregator/init \
  -H "Content-Type: application/json" \
  -d '{"vpp_nodes": [10, 15, 18, 20, 25]}'
```

### 单时间步计算
```bash
curl -X POST http://localhost:5123/xxaqy-api/aggregator/solve/single \
  -H "Content-Type: application/json" \
  -d '{
    "vpp_nodes": [10, 15, 18, 20, 25],
    "p_inj_max": [40000, 10000, 40000, 4000, 4000],
    "p_abs_max": [1000, 1000, 1000, 1000, 1000]
  }'
```

### 多时间步计算
```bash
curl -X POST http://localhost:5123/xxaqy-api/aggregator/solve \
  -H "Content-Type: application/json" \
  -d '{
    "vpp_nodes": [10, 15, 18, 20, 25],
    "p_inj_max_profile": [[40000, 10000, 40000, 4000, 4000], [38000, 9500, 38000, 3800, 3800]],
    "p_abs_max_profile": [[1000, 1000, 1000, 1000, 1000], [1100, 1100, 1100, 1100, 1100]]
  }'
```

---

## Python 客户端示例

```python
import requests
import numpy as np

BASE_URL = "http://localhost:5123"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/xxaqy-api/health")
print(response.json())

# 2. 初始化聚合器
response = requests.post(
    f"{BASE_URL}/xxaqy-api/aggregator/init",
    json={"vpp_nodes": [10, 15, 18, 20, 25]}
)
print(response.json())

# 3. 单时间步计算
response = requests.post(
    f"{BASE_URL}/xxaqy-api/aggregator/solve/single",
    json={
        "vpp_nodes": [10, 15, 18, 20, 25],
        "p_inj_max": [40000, 10000, 40000, 4000, 4000],
        "p_abs_max": [1000, 1000, 1000, 1000, 1000]
    }
)
result = response.json()
print(f"最大安全注入功率: {result['data']['net_inj_max']} kW")
print(f"最大安全吸收功率: {result['data']['net_abs_max']} kW")

# 4. 多时间步计算（24小时）
T = 24
num_vpp = 5
base_p_max = np.array([40000, 10000, 40000, 4000, 4000])
base_p_min_abs = np.array([1000, 1000, 1000, 1000, 1000])

p_inj_max_profile = []
p_abs_max_profile = []

for t in range(T):
    hour = t + 1
    scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi)
    p_inj_max_profile.append((base_p_max * scale).tolist())
    p_abs_max_profile.append((base_p_min_abs * (1 + 0.5 * scale)).tolist())

response = requests.post(
    f"{BASE_URL}/xxaqy-api/aggregator/solve",
    json={
        "vpp_nodes": [10, 15, 18, 20, 25],
        "p_inj_max_profile": p_inj_max_profile,
        "p_abs_max_profile": p_abs_max_profile,
        "q_ratio": 0.5
    }
)
results = response.json()
print(f"24小时聚合结果: {results['data']}")
```

---

## 错误处理

所有接口在出错时都会返回以下格式的错误响应：

```json
{
  "success": false,
  "error": "错误描述",
  "traceback": "详细堆栈信息（仅在开发模式）"
}
```

常见错误码：
- `400`: 请求参数错误
- `500`: 服务器内部错误
- `404`: 接口不存在
