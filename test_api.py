"""
简单的 API 测试脚本
用于快速测试 Flask API 功能
"""
import requests
import numpy as np
import json

BASE_URL = "http://localhost:5123"


def test_health_check():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试: 健康检查")
    print("=" * 50)
    try:
        response = requests.get(f"{BASE_URL}/xxaqy-api/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("✓ 健康检查通过\n")
        return True
    except Exception as e:
        print(f"✗ 健康检查失败: {e}\n")
        return False


def test_build_matrices():
    """测试构建矩阵接口"""
    print("=" * 50)
    print("测试: 构建约束矩阵")
    print("=" * 50)
    try:
        response = requests.post(
            f"{BASE_URL}/xxaqy-api/matrices/build",
            json={"verbose": False, "vpp_nodes": [10, 15, 18, 20, 25]}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        if result.get('success'):
            print(f"✓ 矩阵构建成功")
            print(f"  A_V 形状: {result['data']['shapes']['A_V']}")
            print(f"  A_I 形状: {result['data']['shapes']['A_I']}")
        else:
            print(f"✗ 矩阵构建失败: {result.get('error')}")
        print()
        return result.get('success', False)
    except Exception as e:
        print(f"✗ 矩阵构建失败: {e}\n")
        return False


def test_init_aggregator():
    """测试初始化聚合器接口"""
    print("=" * 50)
    print("测试: 初始化聚合器")
    print("=" * 50)
    try:
        response = requests.post(
            f"{BASE_URL}/xxaqy-api/aggregator/init",
            json={"vpp_nodes": [10, 15, 18, 20, 25]}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        if result.get('success'):
            print(f"✓ 聚合器初始化成功")
            print(f"  VPP 节点: {result['vpp_nodes']}")
        else:
            print(f"✗ 聚合器初始化失败: {result.get('error')}")
        print()
        return result.get('success', False)
    except Exception as e:
        print(f"✗ 聚合器初始化失败: {e}\n")
        return False


def test_solve_single():
    """测试单时间步计算接口"""
    print("=" * 50)
    print("测试: 单时间步聚合计算")
    print("=" * 50)
    try:
        response = requests.post(
            f"{BASE_URL}/xxaqy-api/aggregator/solve/single",
            json={
                "vpp_nodes": [10, 15, 18, 20, 25],
                "p_inj_max": [40000, 10000, 40000, 4000, 4000],
                "p_abs_max": [1000, 1000, 1000, 1000, 1000],
                "q_ratio": 0.5
            }
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        if result.get('success'):
            data = result['data']
            print(f"✓ 计算成功")
            print(f"  最大安全注入功率: {data['net_inj_max']:.2f} kW")
            print(f"  最大安全吸收功率: {data['net_abs_max']:.2f} kW")
            print(f"  物理注入上限: {data['phy_inj_sum']:.2f} kW")
            print(f"  物理吸收上限: {data['phy_abs_sum']:.2f} kW")
        else:
            print(f"✗ 计算失败: {result.get('error')}")
        print()
        return result.get('success', False)
    except Exception as e:
        print(f"✗ 计算失败: {e}\n")
        return False


def test_solve_multiple():
    """测试多时间步计算接口"""
    print("=" * 50)
    print("测试: 多时间步聚合计算 (24小时)")
    print("=" * 50)
    try:
        # 生成24小时数据
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
                "q_ratio": 0.5,
                "use_cache": True
            }
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        if result.get('success'):
            data = result['data']
            print(f"✓ 计算成功")
            print(f"  时间步数: {result['metadata']['num_time_steps']}")
            print(f"  第1小时最大安全注入: {data['net_inj_max'][0]:.2f} kW")
            print(f"  第1小时最大安全吸收: {data['net_abs_max'][0]:.2f} kW")
            print(f"  第12小时最大安全注入: {data['net_inj_max'][11]:.2f} kW")
            print(f"  第12小时最大安全吸收: {data['net_abs_max'][11]:.2f} kW")
        else:
            print(f"✗ 计算失败: {result.get('error')}")
        print()
        return result.get('success', False)
    except Exception as e:
        print(f"✗ 计算失败: {e}\n")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("DERs Safe Aggregation API 测试")
    print("=" * 50 + "\n")

    results = []

    # 运行测试
    results.append(("健康检查", test_health_check()))
    results.append(("构建矩阵", test_build_matrices()))
    results.append(("初始化聚合器", test_init_aggregator()))
    results.append(("单时间步计算", test_solve_single()))
    results.append(("多时间步计算", test_solve_multiple()))

    # 汇总结果
    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查服务是否正常运行")


if __name__ == "__main__":
    main()
