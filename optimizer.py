# optimizer.py
import gurobipy as gp
from gurobipy import GRB


class DSOptimizer:
    """配电网优化器基类（定义统一接口）"""

    def __init__(self, data, pv_params, ess_params, ev_params=None, ac_params=None, **kwargs):
        self.data = data
        self.pv_params = pv_params
        self.ess_params = ess_params
        self.ev_params = ev_params if ev_params is not None else {}
        self.ac_params = ac_params if ac_params is not None else {}
        self.kwargs = kwargs
        self.model = None
        self.vars = {}
        self.results = None
        self.objective_type = self.kwargs.get('objective_type', 'cost_min')

    def build_model(self):
        """构建优化模型（子类实现）"""
        raise NotImplementedError

    def solve(self):
        """求解模型"""
        if self.model is None:
            self.build_model()
        self.model.optimize()
        self._process_results()
        return self.results

    def _process_results(self):
        """处理求解结果（子类实现）"""
        raise NotImplementedError


class SingleAreaOptimizer(DSOptimizer):
    """单区域优化器（集中式控制）"""

    def build_model(self):
        data = self.data
        buses = data['buses']
        lines = data['lines']
        slack_bus = data['slack_bus']

        # 参数提取
        tso_p_set = self.kwargs.get('tso_p_set', 1.5)
        price_grid = self.kwargs.get('price_grid', 50.0)
        Vmin_pu = self.kwargs.get('Vmin_pu', 0.95)
        Vmax_pu = self.kwargs.get('Vmax_pu', 1.05)
        flow_bound = self.kwargs.get('flow_bound', 50.0)
        shed_penalty = self.kwargs.get('shed_penalty', 1e5)

        # 创建模型
        self.model = gp.Model("SingleArea_DSO")
        if not self.kwargs.get('log_to_console', True):
            self.model.Params.OutputFlag = 0

        # 全局变量
        self.vars['V'] = self.model.addVars(buses, lb=Vmin_pu, ub=Vmax_pu, name="Voltage_pu")
        self.vars['P_ij'] = self.model.addVars([(i, j) for i, j, _, _ in lines],
                                               lb=-flow_bound, ub=flow_bound, name="ActiveFlow_MW")
        self.vars['Q_ij'] = self.model.addVars([(i, j) for i, j, _, _ in lines],
                                               lb=-flow_bound, ub=flow_bound, name="ReactiveFlow_MVAr")
        self.vars['P_grid'] = self.model.addVar(lb=0.0, ub=flow_bound, name="GridActivePower_MW")
        self.vars['Q_grid'] = self.model.addVar(lb=-flow_bound, ub=flow_bound, name="GridReactivePower_MVAr")
        self.vars['LS_P'] = self.model.addVars(buses, lb=0.0, name="LoadShed_Active_MW")
        self.vars['LS_Q'] = self.model.addVars(buses, lb=0.0, name="LoadShed_Reactive_MVAr")

        # DER变量
        self.vars['ders'] = {}
        # PV
        for bus, params in self.pv_params.items():
            self.vars['ders'][f"PV_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pmax'],
                                                               name=f"PV_Bus{bus}_Active_MW")
        # ESS
        for bus, params in self.ess_params.items():
            self.vars['ders'][f"ESS_Charge_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pch_max'],
                                                                       name=f"ESS_Bus{bus}_Charge_MW")
            self.vars['ders'][f"ESS_Discharge_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pdis_max'],
                                                                          name=f"ESS_Bus{bus}_Discharge_MW")
            self.vars['ders'][f"ESS_SOC_{bus}"] = self.model.addVar(lb=params['Emin'], ub=params['Emax'],
                                                                    name=f"ESS_Bus{bus}_SOC_MWh")
            # ESS充放电互斥
            self.model.addConstr(
                self.vars['ders'][f"ESS_Charge_{bus}"] + self.vars['ders'][f"ESS_Discharge_{bus}"]
                <= params['Pch_max'],
                name=f"ESS_Bus{bus}_Mutex"
            )
            # SOC平衡
            self.model.addConstr(
                self.vars['ders'][f"ESS_SOC_{bus}"] == params['E0'] +
                self.vars['ders'][f"ESS_Charge_{bus}"] * params['eta_ch'] -
                self.vars['ders'][f"ESS_Discharge_{bus}"] / params['eta_dis'],
                name=f"ESS_Bus{bus}_SOC"
            )
        # 充电桩(EV)
        for bus, params in self.ev_params.items():
            self.vars['ders'][f"EV_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pmax'],
                                                              name=f"EV_Bus{bus}_Active_MW")
        # 空调(AC)
        for bus, params in self.ac_params.items():
            self.vars['ders'][f"AC_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pmax'],
                                                              name=f"AC_Bus{bus}_Active_MW")

        # 约束
        # 平衡母线电压
        self.model.addConstr(self.vars['V'][slack_bus] == 1.0, name="Slack_Voltage")
        # 电压降落
        for i, j, r_pu, x_pu in lines:
            self.model.addConstr(
                self.vars['V'][j] == self.vars['V'][i] - 2 * (
                            r_pu * self.vars['P_ij'][(i, j)] + x_pu * self.vars['Q_ij'][(i, j)]),
                name=f"VoltageDrop_{i}_{j}"
            )
        # 有功平衡
        total_der_p = gp.quicksum(
            var for name, var in self.vars['ders'].items()
            if "PV" in name or "Discharge" in name or "EV" in name or "AC" in name
        ) - gp.quicksum(
            var for name, var in self.vars['ders'].items() if "Charge" in name
        )
        total_load_p = gp.quicksum(data['Pd_MW'][bus] for bus in buses)
        total_ls_p = gp.quicksum(self.vars['LS_P'][bus] for bus in buses)
        slack_outflow_p = gp.quicksum(self.vars['P_ij'][(slack_bus, c)] for i, c, _, _ in lines if i == slack_bus)
        self.model.addConstr(self.vars['P_grid'] == slack_outflow_p, name="Slack_Active_Balance")
        self.model.addConstr(self.vars['P_grid'] + total_der_p == total_load_p - total_ls_p,
                             name="Global_Active_Balance")
        # 无功平衡
        total_load_q = gp.quicksum(data['Qd_MVAr'][bus] for bus in buses)
        total_ls_q = gp.quicksum(self.vars['LS_Q'][bus] for bus in buses)
        slack_outflow_q = gp.quicksum(self.vars['Q_ij'][(slack_bus, c)] for i, c, _, _ in lines if i == slack_bus)
        self.model.addConstr(self.vars['Q_grid'] == slack_outflow_q, name="Slack_Reactive_Balance")
        self.model.addConstr(self.vars['Q_grid'] == total_load_q - total_ls_q, name="Global_Reactive_Balance")
        # TSO指令跟踪
        self.model.addConstr(self.vars['P_grid'] >= tso_p_set - 0.001, name="TSO_Lower")
        self.model.addConstr(self.vars['P_grid'] <= tso_p_set + 0.001, name="TSO_Upper")

        # 构建目标函数
        obj = self._build_objective(price_grid, shed_penalty, total_ls_p, total_ls_q)
        self.model.setObjective(obj, GRB.MINIMIZE)

    def _build_objective(self, price_grid, shed_penalty, total_ls_p, total_ls_q):
        """构建不同类型的目标函数"""
        if self.objective_type == 'cost_min':
            return self._build_cost_objective(price_grid, shed_penalty, total_ls_p, total_ls_q)
        elif self.objective_type == 'voltage_deviation_min':
            return self._build_voltage_deviation_objective(price_grid)
        elif self.objective_type == 'current_deviation_min':
            return self._build_current_deviation_objective(price_grid)
        elif self.objective_type == 'pv_priority':
            return self._build_pv_priority_objective(price_grid)
        elif self.objective_type == 'ess_priority':
            return self._build_ess_priority_objective(price_grid)
        elif self.objective_type == 'ev_priority':
            return self._build_ev_priority_objective(price_grid)
        elif self.objective_type == 'ac_priority':
            return self._build_ac_priority_objective(price_grid)
        else:
            raise ValueError(f"不支持的目标类型: {self.objective_type}")

    def _build_cost_objective(self, price_grid, shed_penalty, total_ls_p, total_ls_q):
        """成本最小目标"""
        obj = price_grid * self.vars['P_grid']
        # PV成本
        for bus, params in self.pv_params.items():
            obj += params['C_double_prime'] * (self.vars['ders'][f"PV_{bus}"] **2)
        # ESS成本
        for bus, params in self.ess_params.items():
            obj += params['C_double_prime'] * (
                    self.vars['ders'][f"ESS_Charge_{bus}"]** 2 +
                    self.vars['ders'][f"ESS_Discharge_{bus}"] **2
            )
        # EV成本
        for bus, params in self.ev_params.items():
            obj += params['C_double_prime'] * (self.vars['ders'][f"EV_{bus}"]** 2)
        # AC成本
        for bus, params in self.ac_params.items():
            obj += params['C_double_prime'] * (self.vars['ders'][f"AC_{bus}"] **2)
        # 负荷削减惩罚
        obj += shed_penalty * (total_ls_p + total_ls_q)
        return obj

    def _build_voltage_deviation_objective(self, price_grid):
        """电压偏差最小目标"""
        # 最小化所有母线电压与1.0pu的偏差平方和
        obj = gp.quicksum((self.vars['V'][bus] - 1.0)** 2 for bus in self.data['buses'])
        # 加入小权重的成本项避免极端解
        obj += 1e-4 * price_grid * self.vars['P_grid']
        return obj

    def _build_current_deviation_objective(self, price_grid):
        """电流偏差最小目标（基于线路潮流）"""
        obj = gp.quicksum(self.vars['P_ij'][(i,j)]**2 + self.vars['Q_ij'][(i,j)]** 2
                         for i,j,_,_ in self.data['lines'])
        # 加入小权重的成本项
        obj += 1e-4 * price_grid * self.vars['P_grid']
        return obj

    def _build_pv_priority_objective(self, price_grid):
        """优先调度光伏（最大化光伏出力）"""
        # 最小化光伏弃用（用负号实现最大化）
        obj = -gp.quicksum(self.vars['ders'][f"PV_{bus}"] for bus in self.pv_params.keys())
        # 加入必要的成本约束
        obj += 1e-2 * price_grid * self.vars['P_grid']
        return obj

    def _build_ess_priority_objective(self, price_grid):
        """优先调度储能（最大化放电）"""
        obj = -gp.quicksum(self.vars['ders'][f"ESS_Discharge_{bus}"] for bus in self.ess_params.keys())
        obj += gp.quicksum(self.vars['ders'][f"ESS_Charge_{bus}"] for bus in self.ess_params.keys())
        # 加入成本约束
        obj += 1e-2 * price_grid * self.vars['P_grid']
        return obj

    def _build_ev_priority_objective(self, price_grid):
        """优先调度充电桩"""
        obj = -gp.quicksum(self.vars['ders'][f"EV_{bus}"] for bus in self.ev_params.keys())
        obj += 1e-2 * price_grid * self.vars['P_grid']
        return obj

    def _build_ac_priority_objective(self, price_grid):
        """优先调度空调"""
        obj = -gp.quicksum(self.vars['ders'][f"AC_{bus}"] for bus in self.ac_params.keys())
        obj += 1e-2 * price_grid * self.vars['P_grid']
        return obj

    def _process_results(self):
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            self.results = None
            return

        data = self.data
        self.results = {
            'status': self.model.Status,
            'objective_value': self.model.objVal,
            'global': {
                'Grid_Power_MW': self.vars['P_grid'].X,
                'Grid_Reactive_MVAr': self.vars['Q_grid'].X,
                'Total_LoadShed_P_MW': sum(self.vars['LS_P'][bus].X for bus in data['buses']),
                'Total_LoadShed_Q_MVAr': sum(self.vars['LS_Q'][bus].X for bus in data['buses']),
                'Voltage_PU': {bus: self.vars['V'][bus].X for bus in data['buses']},
                'Branch_ActiveFlow_MW': {(i, j): self.vars['P_ij'][(i, j)].X for i, j, _, _ in data['lines']},
                'Branch_ReactiveFlow_MVAr': {(i, j): self.vars['Q_ij'][(i, j)].X for i, j, _, _ in data['lines']}
            },
            'ders': {name: var.X for name, var in self.vars['ders'].items()}
        }


class MultiAreaOptimizer(DSOptimizer):
    """多区域优化器（层级控制）"""

    def __init__(self, data, pv_params, ess_params, control_areas, vder_params,
                 ev_params=None, ac_params=None, **kwargs):
        super().__init__(data, pv_params, ess_params, ev_params, ac_params,** kwargs)
        self.control_areas = control_areas
        self.vder_params = vder_params
        self.bus_to_area = {bus: aid for aid, info in control_areas.items() for bus in info['buses']}

    def build_model(self):
        data = self.data
        buses = data['buses']
        lines = data['lines']
        slack_bus = data['slack_bus']

        # 参数提取
        tso_p_set = self.kwargs.get('tso_p_set', 1.5)
        price_grid = self.kwargs.get('price_grid', 50.0)
        Vmin_pu = self.kwargs.get('Vmin_pu', 0.95)
        Vmax_pu = self.kwargs.get('Vmax_pu', 1.05)
        flow_bound = self.kwargs.get('flow_bound', 50.0)
        shed_penalty = self.kwargs.get('shed_penalty', 1e5)

        # 创建模型
        self.model = gp.Model("MultiArea_DSO")
        if not self.kwargs.get('log_to_console', True):
            self.model.Params.OutputFlag = 0

        # 全局变量
        self.vars['V'] = self.model.addVars(buses, lb=Vmin_pu, ub=Vmax_pu, name="Voltage_pu")
        self.vars['P_ij'] = self.model.addVars([(i, j) for i, j, _, _ in lines],
                                               lb=-flow_bound, ub=flow_bound, name="ActiveFlow_MW")
        self.vars['Q_ij'] = self.model.addVars([(i, j) for i, j, _, _ in lines],
                                               lb=-flow_bound, ub=flow_bound, name="ReactiveFlow_MVAr")
        self.vars['P_grid'] = self.model.addVar(lb=0.0, ub=flow_bound, name="GridActivePower_MW")
        self.vars['Q_grid'] = self.model.addVar(lb=-flow_bound, ub=flow_bound, name="GridReactivePower_MVAr")
        self.vars['LS_P'] = self.model.addVars(buses, lb=0.0, name="LoadShed_Active_MW")
        self.vars['LS_Q'] = self.model.addVars(buses, lb=0.0, name="LoadShed_Reactive_MVAr")

        # 区域变量
        self.vars['areas'] = {}
        for aid in self.control_areas.keys():
            self.vars['areas'][aid] = {
                'P_exchange': self.model.addVar(lb=-flow_bound, ub=flow_bound,
                                               name=f"Area{aid}_Exchange_P"),
                'Q_exchange': self.model.addVar(lb=-flow_bound, ub=flow_bound,
                                               name=f"Area{aid}_Exchange_Q")
            }

        # DER变量（与单区域类似，略作调整以适配多区域）
        self.vars['ders'] = {}
        # PV
        for bus, params in self.pv_params.items():
            self.vars['ders'][f"PV_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pmax'],
                                                               name=f"PV_Bus{bus}_Active_MW")
        # ESS
        for bus, params in self.ess_params.items():
            self.vars['ders'][f"ESS_Charge_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pch_max'],
                                                                       name=f"ESS_Bus{bus}_Charge_MW")
            self.vars['ders'][f"ESS_Discharge_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pdis_max'],
                                                                          name=f"ESS_Bus{bus}_Discharge_MW")
            self.vars['ders'][f"ESS_SOC_{bus}"] = self.model.addVar(lb=params['Emin'], ub=params['Emax'],
                                                                    name=f"ESS_Bus{bus}_SOC_MWh")
            self.model.addConstr(
                self.vars['ders'][f"ESS_Charge_{bus}"] + self.vars['ders'][f"ESS_Discharge_{bus}"]
                <= params['Pch_max'],
                name=f"ESS_Bus{bus}_Mutex"
            )
            self.model.addConstr(
                self.vars['ders'][f"ESS_SOC_{bus}"] == params['E0'] +
                self.vars['ders'][f"ESS_Charge_{bus}"] * params['eta_ch'] -
                self.vars['ders'][f"ESS_Discharge_{bus}"] / params['eta_dis'],
                name=f"ESS_Bus{bus}_SOC"
            )
        # EV和AC变量
        for bus, params in self.ev_params.items():
            self.vars['ders'][f"EV_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pmax'],
                                                              name=f"EV_Bus{bus}_Active_MW")
        for bus, params in self.ac_params.items():
            self.vars['ders'][f"AC_{bus}"] = self.model.addVar(lb=0.0, ub=params['Pmax'],
                                                              name=f"AC_Bus{bus}_Active_MW")

        # 多区域约束（简化版）
        self.model.addConstr(self.vars['V'][slack_bus] == 1.0, name="Slack_Voltage")
        for i, j, r_pu, x_pu in lines:
            self.model.addConstr(
                self.vars['V'][j] == self.vars['V'][i] - 2 * (
                            r_pu * self.vars['P_ij'][(i, j)] + x_pu * self.vars['Q_ij'][(i, j)]),
                name=f"VoltageDrop_{i}_{j}"
            )

        # 目标函数（复用单区域的目标函数构建逻辑）
        total_ls_p = gp.quicksum(self.vars['LS_P'][bus] for bus in buses)
        total_ls_q = gp.quicksum(self.vars['LS_Q'][bus] for bus in buses)
        obj = self._build_objective(price_grid, shed_penalty, total_ls_p, total_ls_q)
        self.model.setObjective(obj, GRB.MINIMIZE)

    # 复用单区域的目标函数构建方法
    _build_objective = SingleAreaOptimizer._build_objective
    _build_cost_objective = SingleAreaOptimizer._build_cost_objective
    _build_voltage_deviation_objective = SingleAreaOptimizer._build_voltage_deviation_objective
    _build_current_deviation_objective = SingleAreaOptimizer._build_current_deviation_objective
    _build_pv_priority_objective = SingleAreaOptimizer._build_pv_priority_objective
    _build_ess_priority_objective = SingleAreaOptimizer._build_ess_priority_objective
    _build_ev_priority_objective = SingleAreaOptimizer._build_ev_priority_objective
    _build_ac_priority_objective = SingleAreaOptimizer._build_ac_priority_objective

    def _process_results(self):
        # 多区域结果处理（简化实现）
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            self.results = None
            return

        data = self.data
        self.results = {
            'status': self.model.Status,
            'objective_value': self.model.objVal,
            'global': {
                'Grid_Power_MW': self.vars['P_grid'].X,
                'Total_LoadShed_P_MW': sum(self.vars['LS_P'][bus].X for bus in data['buses']),
                'Voltage_PU': {bus: self.vars['V'][bus].X for bus in data['buses']}
            },
            'areas': {},
            'ders': {name: var.X for name, var in self.vars['ders'].items()}
        }

        # 填充区域结果
        for aid in self.control_areas:
            self.results['areas'][aid] = {
                'Exchange_P': self.vars['areas'][aid]['P_exchange'].X,
                'LoadShed_P_MW': sum(self.vars['LS_P'][bus].X for bus in self.control_areas[aid]['buses']),
                'DER_Outputs': {k: v for k, v in self.results['ders'].items()
                               if any(str(bus) in k for bus in self.control_areas[aid]['buses'])}
            }
