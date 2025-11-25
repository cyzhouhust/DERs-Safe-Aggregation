## 输入说明

### 1. 配置参数（main.py 中 config 字典）
- `use_multi_area`: 优化模式切换，`True`为多区域模式，`False`为单区域模式（默认：`False`）
- `tso_p_set`: TSO有功指令，单位：MW（默认：`1.5`）
- `price_grid`: 购电单价，单位：元/MWh（默认：`50.0`）
- `log_to_console`: 是否显示求解日志（默认：`True`）
- `objective_type`: 目标函数类型，可选值：
  - `cost_min`: 成本最小化（默认）
  - `voltage_deviation_min`: 电压偏差最小化
  - `current_deviation_min`: 电流偏差最小化
  - `pv_priority`: 光伏优先
  - `ess_priority`: 储能优先
  - `ev_priority`: 电动汽车优先
  - `ac_priority`: 空调优先

### 2. 电网数据（data_loader.py）
- **母线数据**
- **支路数据**
- **控制区域划分**：多区域模式下的区域配置，包括：
  - 区域包含的母线
  - 父区域关联
  - 接口母线
  - 包含的DER设备

### 3. DER参数（data_loader.py 中 get_der_params()）
- **光伏（PV）**：最大出力（Pmax）、成本系数（C_double_prime）
- **储能（ESS）**：充放电功率限制、效率、SOC范围、初始SOC、成本系数
- **电动汽车（EV）**：最大充电功率、成本系数
- **空调（AC）**：最大功率、成本系数
- **VDER参数**：多区域模式下的虚拟DER成本系数

## 输出说明

### 1. 控制台输出
- 优化结果摘要：
  - 总目标函数值
  - 电网购电功率（单位：MW）
  - 总有功负荷削减（单位：MW）

### 2. 可视化结果
1. **电压分布**：
   - 所有母线的电压标幺值（pu）
   - 显示电压上下限（0.95pu和1.05pu）参考线
   - 区分单区域/多区域模式

2. **DER出力**：
   - 各类DER（PV、ESS充放电、EV、AC）的有功功率输出（单位：MW）
   - 多区域模式下显示区域标识
   - 忽略极小值（<1e-6 MW）的出力

3. **负荷削减（多区域模式）**：
   - 各控制区域的有功负荷削减对比（单位：MW）

### 3. 优化结果数据结构（results）
- `objective_value`: 目标函数值
- `global`: 全局结果
  - `Voltage_PU`: 各母线电压标幺值
  - `Grid_Power_MW`: 电网购电功率
  - `Total_LoadShed_P_MW`: 总有功负荷削减
- `ders` (单区域) / `areas` (多区域): DER出力及区域结果详情
  - DER出力值（PV、ESS充放电、EV、AC）
  - 区域负荷削减（多区域）

## 使用方法
1. 调整`main.py`中的`config`参数配置优化模式和目标
2. 运行`main.py`执行优化计算
3. 查看输出可视化图表

