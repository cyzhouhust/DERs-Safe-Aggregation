"""
Flask API for DERs Safe Aggregation
封装所有算法为 REST 接口
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback
from matrices import build_matrices
from agg4 import VPPAggregator

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储聚合器实例（可选，用于缓存）
aggregator_cache = {}


@app.route('/xxaqy-api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'message': 'DERs Safe Aggregation API is running'
    })


@app.route('/xxaqy-api/matrices/build', methods=['POST'])
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


@app.route('/xxaqy-api/aggregator/init', methods=['POST'])
def init_aggregator():
    """
    初始化 VPP 聚合器
    请求体:
    {
        "vpp_nodes": [10, 15, 18, 20, 25]
    }
    """
    try:
        data = request.get_json() or {}
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
        
        if not isinstance(vpp_nodes, list) or len(vpp_nodes) == 0:
            return jsonify({
                'success': False,
                'error': 'vpp_nodes must be a non-empty list'
            }), 400
        
        # 创建聚合器实例
        aggregator = VPPAggregator(vpp_nodes)
        
        # 生成缓存键
        cache_key = ','.join(map(str, sorted(vpp_nodes)))
        aggregator_cache[cache_key] = aggregator
        
        return jsonify({
            'success': True,
            'message': 'Aggregator initialized successfully',
            'vpp_nodes': vpp_nodes,
            'cache_key': cache_key
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/xxaqy-api/aggregator/solve', methods=['POST'])
def solve_dispatch():
    """
    执行聚合优化计算
    请求体:
    {
        "vpp_nodes": [10, 15, 18, 20, 25],
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
        
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
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
        
        if num_vpp != len(vpp_nodes):
            return jsonify({
                'success': False,
                'error': f'Number of nodes in profiles ({num_vpp}) does not match vpp_nodes length ({len(vpp_nodes)})'
            }), 400
        
        # 获取或创建聚合器
        cache_key = ','.join(map(str, sorted(vpp_nodes)))
        if use_cache and cache_key in aggregator_cache:
            aggregator = aggregator_cache[cache_key]
        else:
            aggregator = VPPAggregator(vpp_nodes)
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
                'vpp_nodes': vpp_nodes,
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


@app.route('/xxaqy-api/aggregator/solve/single', methods=['POST'])
def solve_single_time_step():
    """
    计算单个时间步的聚合能力
    请求体:
    {
        "vpp_nodes": [10, 15, 18, 20, 25],
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
        
        vpp_nodes = data.get('vpp_nodes', [10, 15, 18, 20, 25])
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
        
        if len(p_inj_max) != len(p_abs_max) or len(p_inj_max) != len(vpp_nodes):
            return jsonify({
                'success': False,
                'error': 'Length mismatch between p_inj_max, p_abs_max, and vpp_nodes'
            }), 400
        
        # 扩展为单时间步的 2D 数组
        p_inj_max_profile = p_inj_max.reshape(1, -1)
        p_abs_max_profile = p_abs_max.reshape(1, -1)
        
        # 获取或创建聚合器
        cache_key = ','.join(map(str, sorted(vpp_nodes)))
        if cache_key in aggregator_cache:
            aggregator = aggregator_cache[cache_key]
        else:
            aggregator = VPPAggregator(vpp_nodes)
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
                'vpp_nodes': vpp_nodes,
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


@app.route('/xxaqy-api/aggregator/clear_cache', methods=['POST'])
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
