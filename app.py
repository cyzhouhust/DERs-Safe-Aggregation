"""
Flask API for DERs Safe Aggregation
封装所有算法为 REST 接口
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback
from scipy.optimize import linprog
from matrices import build_matrices
from agg4 import VPPAggregator
from agg5 import solve_snapshot_capacity, RESOURCE_GROUPS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储聚合器实例（可选，用于缓存）
aggregator_cache = {}


@app.route('/xxaqy-api/agg4/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'message': 'DERs Safe Aggregation API (agg4) is running'
    })


@app.route('/xxaqy-api/agg4/matrices/build', methods=['POST'])
def build_matrices_api():
    """
    构建约束矩阵接口
    请求体（可选）:
    {
        "verbose": false,
        "vpp_nodes": [10, 15, 18, 20, 25]  # 可选，用于指定VPP节点
    }
    """
    try:
        data = request.get_json() or {}
        verbose = data.get('verbose', False)
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
        
        # 调用 build_matrices 函数
        result = build_matrices(verbose=verbose)
        
        # 解包结果
        A_V, b_V, A_I, b_I, R_hat, X_hat, R_I, X_I, Pd, Qd, \
        C_MI_load_neg, C_MI_load_pos, C_mV_load, C_MV_load = result
        
        # 转换为可序列化的格式
        response = {
            'success': True,
            'data': {
                'A_V': A_V.tolist(),
                'b_V': b_V.tolist(),
                'A_I': A_I.tolist(),
                'b_I': b_I.tolist(),
                'R_hat': R_hat.tolist(),
                'X_hat': X_hat.tolist(),
                'R_I': R_I.tolist(),
                'X_I': X_I.tolist(),
                'Pd': Pd.flatten().tolist(),
                'Qd': Qd.flatten().tolist(),
                'C_MI_load_neg': C_MI_load_neg.flatten().tolist(),
                'C_MI_load_pos': C_MI_load_pos.flatten().tolist(),
                'C_mV_load': C_mV_load.flatten().tolist(),
                'C_MV_load': C_MV_load.flatten().tolist(),
                'shapes': {
                    'A_V': list(A_V.shape),
                    'b_V': list(b_V.shape),
                    'A_I': list(A_I.shape),
                    'b_I': list(b_I.shape),
                    'R_hat': list(R_hat.shape),
                    'X_hat': list(X_hat.shape),
                    'R_I': list(R_I.shape),
                    'X_I': list(X_I.shape),
                }
            },
            'vpp_nodes': vpp_nodes
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg4/aggregator/init', methods=['POST'])
def init_aggregator():
    """
    初始化 VPP 聚合器
    请求体:
    {
        "vpp_config": {
            "Type A (Solar)": [10, 15],
            "Type B (Wind)": [18],
            "Type C (ESS)": [20],
            "Type D (Gas)": [25]
        }
    }
    或者使用简化格式（自动分组）:
    {
        "vpp_nodes": [10, 15, 18, 20, 25]
    }
    """
    try:
        data = request.get_json() or {}
        
        # 支持两种输入格式
        if 'vpp_config' in data:
            vpp_config = data['vpp_config']
            if not isinstance(vpp_config, dict):
                return jsonify({
                    'success': False,
                    'error': 'vpp_config must be a dictionary'
                }), 400
        elif 'vpp_nodes' in data:
            vpp_nodes = data['vpp_nodes']
            if not isinstance(vpp_nodes, list) or len(vpp_nodes) == 0:
                return jsonify({
                    'success': False,
                    'error': 'vpp_nodes must be a non-empty list'
                }), 400
            # 自动创建默认分组
            vpp_config = {
                'Type A (Solar)': [10, 15],
                'Type B (Wind)': [18],
                'Type C (ESS)': [20],
                'Type D (Gas)': [25]
            }
        else:
            # 使用默认配置
            vpp_config = {
                'Type A (Solar)': [10, 15],
                'Type B (Wind)': [18],
                'Type C (ESS)': [20],
                'Type D (Gas)': [25]
            }
        
        # 创建聚合器实例
        aggregator = VPPAggregator(vpp_config)
        
        # 生成缓存键（基于所有节点的排序）
        all_nodes = []
        for nodes in vpp_config.values():
            all_nodes.extend(nodes)
        cache_key = ','.join(map(str, sorted(all_nodes)))
        aggregator_cache[cache_key] = aggregator
        
        return jsonify({
            'success': True,
            'message': 'Aggregator initialized successfully',
            'vpp_config': vpp_config,
            'cache_key': cache_key
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg4/aggregator/solve', methods=['POST'])
def solve_dispatch():
    """
    执行聚合优化计算
    请求体:
    {
        "vpp_config": {  # 可选，使用字典格式
            "Type A (Solar)": [10, 15],
            "Type B (Wind)": [18],
            "Type C (ESS)": [20],
            "Type D (Gas)": [25]
        },
        或
        "vpp_nodes": [10, 15, 18, 20, 25],  # 可选，使用列表格式（自动分组）
        "p_inj_max_profile": [[40000, 10000, 40000, 4000, 4000], ...],  # T x num_vpp
        "p_abs_max_profile": [[1000, 1000, 1000, 1000, 1000], ...],     # T x num_vpp
        "q_ratio": 0.5,  # 可选，默认0.5
        "use_cache": true  # 可选，是否使用缓存的聚合器实例
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        p_inj_max_profile = data.get('p_inj_max_profile')
        p_abs_max_profile = data.get('p_abs_max_profile')
        q_ratio = data.get('q_ratio', 0.5)
        use_cache = data.get('use_cache', True)
        
        # 验证输入
        if p_inj_max_profile is None or p_abs_max_profile is None:
            return jsonify({
                'success': False,
                'error': 'p_inj_max_profile and p_abs_max_profile are required'
            }), 400
        
        # 转换为 numpy 数组
        p_inj_max_profile = np.array(p_inj_max_profile)
        p_abs_max_profile = np.array(p_abs_max_profile)
        
        # 验证维度
        if len(p_inj_max_profile.shape) != 2 or len(p_abs_max_profile.shape) != 2:
            return jsonify({
                'success': False,
                'error': 'p_inj_max_profile and p_abs_max_profile must be 2D arrays'
            }), 400
        
        if p_inj_max_profile.shape != p_abs_max_profile.shape:
            return jsonify({
                'success': False,
                'error': 'p_inj_max_profile and p_abs_max_profile must have the same shape'
            }), 400
        
        T, num_vpp = p_inj_max_profile.shape
        
        # 处理 vpp_config 或 vpp_nodes
        if 'vpp_config' in data:
            vpp_config = data['vpp_config']
            if not isinstance(vpp_config, dict):
                return jsonify({
                    'success': False,
                    'error': 'vpp_config must be a dictionary'
                }), 400
            # 从配置中提取所有节点
            all_nodes = []
            for nodes in vpp_config.values():
                all_nodes.extend(nodes)
        elif 'vpp_nodes' in data:
            vpp_nodes = data['vpp_nodes']
            if not isinstance(vpp_nodes, list):
                return jsonify({
                    'success': False,
                    'error': 'vpp_nodes must be a list'
                }), 400
            # 自动创建默认分组
            vpp_config = {
                'Type A (Solar)': [10, 15],
                'Type B (Wind)': [18],
                'Type C (ESS)': [20],
                'Type D (Gas)': [25]
            }
            all_nodes = vpp_nodes
        else:
            # 使用默认配置
            vpp_config = {
                'Type A (Solar)': [10, 15],
                'Type B (Wind)': [18],
                'Type C (ESS)': [20],
                'Type D (Gas)': [25]
            }
            all_nodes = [10, 15, 18, 20, 25]
        
        if num_vpp != len(all_nodes):
            return jsonify({
                'success': False,
                'error': f'Number of nodes in profiles ({num_vpp}) does not match vpp_config nodes ({len(all_nodes)})'
            }), 400
        
        # 获取或创建聚合器
        cache_key = ','.join(map(str, sorted(all_nodes)))
        if use_cache and cache_key in aggregator_cache:
            aggregator = aggregator_cache[cache_key]
        else:
            aggregator = VPPAggregator(vpp_config)
            if use_cache:
                aggregator_cache[cache_key] = aggregator
        
        # 执行优化
        results = aggregator.solve_dispatch(
            p_inj_max_profile,
            p_abs_max_profile,
            q_ratio=q_ratio
        )
        
        if results is None:
            return jsonify({
                'success': False,
                'error': 'Optimization failed: network matrices not initialized'
            }), 500
        
        # 转换为可序列化的格式
        response = {
            'success': True,
            'data': {
                'net_inj_max': results['net_inj_max'].tolist(),
                'net_abs_max': results['net_abs_max'].tolist(),
                'phy_inj_sum': results['phy_inj_sum'].tolist(),
                'phy_abs_sum': results['phy_abs_sum'].tolist()
            },
            'metadata': {
                'vpp_config': vpp_config,
                'num_time_steps': T,
                'num_vpp_nodes': num_vpp,
                'q_ratio': q_ratio
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg4/aggregator/solve/single', methods=['POST'])
def solve_single_time_step():
    """
    计算单个时间步的聚合能力
    请求体:
    {
        "vpp_config": {  # 可选
            "Type A (Solar)": [10, 15],
            "Type B (Wind)": [18],
            "Type C (ESS)": [20],
            "Type D (Gas)": [25]
        },
        或
        "vpp_nodes": [10, 15, 18, 20, 25],  # 可选
        "p_inj_max": [40000, 10000, 40000, 4000, 4000],  # 当前时刻各节点最大注入功率
        "p_abs_max": [1000, 1000, 1000, 1000, 1000],    # 当前时刻各节点最大吸收功率（正值）
        "q_ratio": 0.5
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        p_inj_max = data.get('p_inj_max')
        p_abs_max = data.get('p_abs_max')
        q_ratio = data.get('q_ratio', 0.5)
        
        # 验证输入
        if p_inj_max is None or p_abs_max is None:
            return jsonify({
                'success': False,
                'error': 'p_inj_max and p_abs_max are required'
            }), 400
        
        # 转换为 numpy 数组
        p_inj_max = np.array(p_inj_max)
        p_abs_max = np.array(p_abs_max)
        
        # 验证维度
        if len(p_inj_max.shape) != 1 or len(p_abs_max.shape) != 1:
            return jsonify({
                'success': False,
                'error': 'p_inj_max and p_abs_max must be 1D arrays'
            }), 400
        
        # 处理 vpp_config 或 vpp_nodes
        if 'vpp_config' in data:
            vpp_config = data['vpp_config']
            if not isinstance(vpp_config, dict):
                return jsonify({
                    'success': False,
                    'error': 'vpp_config must be a dictionary'
                }), 400
            all_nodes = []
            for nodes in vpp_config.values():
                all_nodes.extend(nodes)
        elif 'vpp_nodes' in data:
            vpp_nodes = data['vpp_nodes']
            vpp_config = {
                'Type A (Solar)': [10, 15],
                'Type B (Wind)': [18],
                'Type C (ESS)': [20],
                'Type D (Gas)': [25]
            }
            all_nodes = vpp_nodes
        else:
            vpp_config = {
                'Type A (Solar)': [10, 15],
                'Type B (Wind)': [18],
                'Type C (ESS)': [20],
                'Type D (Gas)': [25]
            }
            all_nodes = [10, 15, 18, 20, 25]
        
        if len(p_inj_max) != len(p_abs_max) or len(p_inj_max) != len(all_nodes):
            return jsonify({
                'success': False,
                'error': 'Length mismatch between p_inj_max, p_abs_max, and vpp nodes'
            }), 400
        
        # 扩展为单时间步的 2D 数组
        p_inj_max_profile = p_inj_max.reshape(1, -1)
        p_abs_max_profile = p_abs_max.reshape(1, -1)
        
        # 获取或创建聚合器
        cache_key = ','.join(map(str, sorted(all_nodes)))
        if cache_key in aggregator_cache:
            aggregator = aggregator_cache[cache_key]
        else:
            aggregator = VPPAggregator(vpp_config)
            aggregator_cache[cache_key] = aggregator
        
        # 执行优化
        results = aggregator.solve_dispatch(
            p_inj_max_profile,
            p_abs_max_profile,
            q_ratio=q_ratio
        )
        
        if results is None:
            return jsonify({
                'success': False,
                'error': 'Optimization failed: network matrices not initialized'
            }), 500
        
        # 返回单个时间步的结果
        response = {
            'success': True,
            'data': {
                'net_inj_max': float(results['net_inj_max'][0]),
                'net_abs_max': float(results['net_abs_max'][0]),
                'phy_inj_sum': float(results['phy_inj_sum'][0]),
                'phy_abs_sum': float(results['phy_abs_sum'][0])
            },
            'metadata': {
                'vpp_config': vpp_config,
                'q_ratio': q_ratio
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg4/aggregator/clear_cache', methods=['POST'])
def clear_cache():
    """清除聚合器缓存"""
    try:
        count = len(aggregator_cache)
        aggregator_cache.clear()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {count} cached aggregator(s)'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/xxaqy-api/agg1/optimize/basic', methods=['POST'])
def optimize_basic():
    """
    基础线性规划优化（agg.py算法）
    计算单个时刻的最大VPP总有功输出
    请求体:
    {
        "vpp_nodes": [10, 15, 18, 20, 25],
        "p_min_list": [-100, -600, -40000, -50000, -5000],
        "p_max_list": [0, 0, 4000, 4000, 2000],
        "q_min_list": [0, 0, 0, 0, 0],  # 可选
        "q_max_list": [0, 0, 0, 0, 0]   # 可选
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
        p_min_list = data.get('p_min_list')
        p_max_list = data.get('p_max_list')
        q_min_list = data.get('q_min_list', [0] * len(vpp_nodes))
        q_max_list = data.get('q_max_list', [0] * len(vpp_nodes))
        
        if p_min_list is None or p_max_list is None:
            return jsonify({
                'success': False,
                'error': 'p_min_list and p_max_list are required'
            }), 400
        
        num_vpp = len(vpp_nodes)
        
        # 构建约束矩阵
        A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
        A_total = A_V
        b_total = b_V
        
        # 定义目标函数
        c = np.hstack([np.ones(num_vpp), np.zeros(num_vpp)])
        
        # 定义边界
        bounds_P = list(zip(p_min_list, p_max_list))
        bounds_Q = list(zip(q_min_list, q_max_list))
        bounds_total = bounds_P + bounds_Q
        
        # 求解
        result = linprog(
            c,
            A_ub=A_total,
            b_ub=b_total,
            bounds=bounds_total,
            method='highs'
        )
        
        if result.success:
            X_opt = result.x
            P_vpp_opt = X_opt[:num_vpp]
            Q_vpp_opt = X_opt[num_vpp:]
            Total_P_max = -result.fun
            
            response = {
                'success': True,
                'data': {
                    'total_p_max': float(Total_P_max),
                    'objective_value': float(result.fun),
                    'p_vpp_opt': P_vpp_opt.tolist(),
                    'q_vpp_opt': Q_vpp_opt.tolist(),
                    'slack': result.slack.tolist() if hasattr(result, 'slack') else None
                },
                'metadata': {
                    'vpp_nodes': vpp_nodes
                }
            }
            return jsonify(response), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Optimization failed: {result.message}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg2/optimize/injection-absorption', methods=['POST'])
def optimize_injection_absorption():
    """
    计算最大注入和最大吸收功率（agg2.py算法）
    使用约定B：吸收为正
    请求体:
    {
        "vpp_nodes": [10, 15, 18, 20, 25],
        "p_inj_min_list": [0, 0, 0, 0, 0],
        "p_inj_max_list": [40000, 10000, 40000, 4000, 4000],
        "q_inj_min_list": [-250, -300, -200, -200, -350],
        "q_inj_max_list": [250, 300, 200, 200, 350]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
        p_inj_min_list = np.array(data.get('p_inj_min_list', [0, 0, 0, 0, 0]))
        p_inj_max_list = np.array(data.get('p_inj_max_list', [40000, 10000, 40000, 4000, 4000]))
        q_inj_min_list = np.array(data.get('q_inj_min_list', [-250, -300, -200, -200, -350]))
        q_inj_max_list = np.array(data.get('q_inj_max_list', [250, 300, 200, 200, 350]))
        
        num_vpp = len(vpp_nodes)
        
        # 获取约束矩阵（约定A）
        A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
        A_total_A = np.vstack([A_V, A_I])
        b_total_A = np.vstack([b_V, b_I]).flatten()
        
        # 转换为约定B
        A_P_A_part = A_total_A[:, :num_vpp]
        A_Q_A_part = A_total_A[:, num_vpp:]
        A_P_B_part = -A_P_A_part
        A_total_B = np.hstack([A_P_B_part, A_Q_A_part])
        b_total_B = b_total_A
        
        # 转换边界到约定B
        P_B_min_list = -p_inj_max_list
        P_B_max_list = -p_inj_min_list
        bounds_P_B = list(zip(P_B_min_list, P_B_max_list))
        bounds_Q_B = list(zip(q_inj_min_list, q_inj_max_list))
        bounds_total_B = bounds_P_B + bounds_Q_B
        
        # 求解最大注入功率
        c_inj = np.hstack([np.ones(num_vpp), np.zeros(num_vpp)])
        result_inj = linprog(c_inj, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total_B, method='highs')
        
        if result_inj.success:
            Total_P_inj_max = -result_inj.fun
            P_B_opt = result_inj.x[:num_vpp]
            Q_opt = result_inj.x[num_vpp:]
            
            response = {
                'success': True,
                'data': {
                    'max_injection': float(Total_P_inj_max),
                    'p_schedule': (-P_B_opt).tolist(),
                    'q_schedule': Q_opt.tolist()
                },
                'metadata': {
                    'vpp_nodes': vpp_nodes,
                    'convention': 'B (absorption as positive)'
                }
            }
            return jsonify(response), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Optimization failed: {result_inj.message}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg3/optimize/24h-rolling', methods=['POST'])
def optimize_24h_rolling():
    """
    24小时滚动优化（agg3.py算法）
    计算动态边界下的注入和吸收能力
    请求体:
    {
        "vpp_nodes": [10, 15, 18, 20, 25],
        "base_p_max": [40000, 10000, 40000, 4000, 4000],
        "base_p_min_abs": [1000, 1000, 1000, 1000, 1000],
        "time_steps": 24  # 可选，默认24
    }
    """
    try:
        data = request.get_json() or {}
        
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
        BASE_P_MAX = np.array(data.get('base_p_max', [40000, 10000, 40000, 4000, 4000]))
        BASE_P_MIN_ABS = np.array(data.get('base_p_min_abs', [1000, 1000, 1000, 1000, 1000]))
        T = data.get('time_steps', 24)
        
        num_vpp = len(vpp_nodes)
        
        # 定义动态边界生成函数
        def generate_der_limits(t):
            hour = t + 1
            scale = 0.5 + 0.5 * np.cos((hour - 12) / 24 * 2 * np.pi)
            P_inj_max_t = BASE_P_MAX * scale
            P_inj_min_t = -BASE_P_MIN_ABS * (1 + 0.5 * scale)
            Q_limit_t = P_inj_max_t * 0.5
            Q_inj_max_t = np.minimum(Q_limit_t, [250, 300, 200, 200, 350])
            Q_inj_min_t = -Q_inj_max_t
            return P_inj_min_t, P_inj_max_t, Q_inj_min_t, Q_inj_max_t
        
        # 存储结果
        max_network_injection = np.zeros(T)
        max_network_absorption = np.zeros(T)
        max_physical_injection = np.zeros(T)
        max_physical_absorption = np.zeros(T)
        
        # 目标函数
        c_inj = np.hstack([np.ones(num_vpp), np.zeros(num_vpp)])
        c_abs = np.hstack([-np.ones(num_vpp), np.zeros(num_vpp)])
        
        # 24小时循环求解
        for t in range(T):
            # 获取动态边界
            P_inj_min_t, P_inj_max_t, Q_inj_min_t, Q_inj_max_t = generate_der_limits(t)
            Q_inj_min_t, Q_inj_max_t = np.zeros_like(Q_inj_min_t), np.zeros_like(Q_inj_max_t)
            
            # 计算物理容量
            max_physical_injection[t] = np.sum(P_inj_max_t)
            max_physical_absorption[t] = -np.sum(P_inj_min_t)
            
            # 转换边界到约定B
            P_B_min_list_t = -P_inj_max_t
            P_B_max_list_t = -P_inj_min_t
            bounds_P_B_t = list(zip(P_B_min_list_t, P_B_max_list_t))
            bounds_Q_B_t = list(zip(Q_inj_min_t, Q_inj_max_t))
            bounds_total_B_t = bounds_P_B_t + bounds_Q_B_t
            
            # 获取约束矩阵
            A_V, b_V, A_I, b_I, _, _, _, _, _, _, _, _, _, _ = build_matrices(verbose=False)
            A_total_A = np.vstack([A_V, A_I])
            b_total_A = np.vstack([b_V, b_I]).flatten()
            A_P_A_part = A_total_A[:, :num_vpp]
            A_Q_A_part = A_total_A[:, num_vpp:]
            A_P_B_part = -A_P_A_part
            A_total_B = np.hstack([A_P_B_part, A_Q_A_part])
            b_total_B = b_total_A
            
            # 求解最大注入
            res_inj = linprog(c_inj, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total_B_t, method='highs')
            max_network_injection[t] = -res_inj.fun if res_inj.success else np.nan
            
            # 求解最大吸收
            res_abs = linprog(c_abs, A_ub=A_total_B, b_ub=b_total_B, bounds=bounds_total_B_t, method='highs')
            max_network_absorption[t] = res_abs.fun if res_abs.success else np.nan
        
        response = {
            'success': True,
            'data': {
                'max_network_injection': max_network_injection.tolist(),
                'max_network_absorption': max_network_absorption.tolist(),
                'max_physical_injection': max_physical_injection.tolist(),
                'max_physical_absorption': max_physical_absorption.tolist()
            },
            'metadata': {
                'vpp_nodes': vpp_nodes,
                'time_steps': T,
                'base_p_max': BASE_P_MAX.tolist(),
                'base_p_min_abs': BASE_P_MIN_ABS.tolist()
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/agg5/capacity/snapshot', methods=['POST'])
def calculate_snapshot_capacity():
    """
    计算快照时间点的资源容量（agg5算法）
    请求体:
    {
        "t_snapshot": 12,  # 时间快照（小时，默认12）
        "resource_groups": {  # 可选，自定义资源分组
            "Type_A": [0, 1],
            "Type_B": [2]
        }
    }
    """
    try:
        data = request.get_json() or {}
        t_snapshot = data.get('t_snapshot', 12)
        
        # 执行计算
        results = solve_snapshot_capacity(t_snapshot=t_snapshot)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': 'Calculation failed'
            }), 500
        
        # 转换为可序列化的格式
        response = {
            'success': True,
            'data': {
                'up_regulation': results['Up'],
                'down_regulation': results['Down']
            },
            'metadata': {
                't_snapshot': t_snapshot,
                'resource_groups': RESOURCE_GROUPS
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # 开发环境配置
    app.run(host='0.0.0.0', port=5123, debug=True)
