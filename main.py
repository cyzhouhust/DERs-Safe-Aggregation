# main.py
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import (
    load_bus_data, load_branch_data, get_control_areas,
    get_der_params, preprocess_data
)
from optimizer import SingleAreaOptimizer, MultiAreaOptimizer


def visualize_results(results, is_multi_area=False, control_areas=None):
    """可视化优化结果"""
    if results is None:
        print("无结果可可视化")
        return

    # 1. 电压分布
    V_series = pd.Series(results['global']['Voltage_PU']).sort_index()
    plt.figure(figsize=(12, 6))
    ax = V_series.plot(marker='o', color='darkblue', linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Vmin=0.95 pu')
    ax.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='Vmax=1.05 pu')
    ax.set_title(f'Voltage Profile Across All Buses ({["Single", "Multi"][is_multi_area]} Area)', fontsize=14)
    ax.set_xlabel('Bus Index', fontsize=12)
    ax.set_ylabel('Voltage (pu)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. DER出力
    der_outputs = []
    der_labels = []
    if is_multi_area:
        for aid, area_res in results['areas'].items():
            for name, val in area_res['DER_Outputs'].items():
                if "SOC" not in name and val > 1e-6:
                    der_labels.append(f"Area{aid}_{name}")
                    der_outputs.append(val)
    else:
        for name, val in results['ders'].items():
            if "SOC" not in name and val > 1e-6:
                der_labels.append(name)
                der_outputs.append(val)

    if der_outputs:
        plt.figure(figsize=(10, 6))
        plt.bar(der_labels, der_outputs, color=['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'])
        plt.title(f'DER Active Power Output ({["Single", "Multi"][is_multi_area]} Area)', fontsize=14)
        plt.xlabel('DER Identifier', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 3. 负荷削减（多区域显示区域对比）
    if is_multi_area and control_areas:
        area_shed = [results['areas'][aid]['LoadShed_P_MW'] for aid in control_areas.keys()]
        plt.figure(figsize=(8, 6))
        plt.bar([f"Area {aid}" for aid in control_areas.keys()], area_shed, color='#ff7f0e')
        plt.title('Load Shedding by Area (Active Power)', fontsize=14)
        plt.xlabel('Control Area', fontsize=12)
        plt.ylabel('Load Shedding (MW)', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    # 配置参数
    config = {
        'use_multi_area': False,  # 切换模式：True=多区域，False=单区域
        'tso_p_set': 1.5,  # TSO有功指令(MW)
        'price_grid': 50.0,  # 购电单价(元/MWh)
        'log_to_console': True,  # 是否显示求解日志
        # 目标函数类型选择
        'objective_type': 'cost_min'  # 可选值: cost_min, voltage_deviation_min, current_deviation_min,
                                     #        pv_priority, ess_priority, ev_priority, ac_priority
    }

    # 加载数据
    bus_data = load_bus_data()
    branch_data = load_branch_data()
    pv_params, ess_params, vder_params, ev_params, ac_params = get_der_params()
    control_areas = get_control_areas()
    data = preprocess_data(bus_data, branch_data)

    # 选择优化器并求解
    if config['use_multi_area']:
        optimizer = MultiAreaOptimizer(
            data=data,
            pv_params=pv_params,
            ess_params=ess_params,
            ev_params=ev_params,
            ac_params=ac_params,
            control_areas=control_areas,
            vder_params=vder_params,
            **config
        )
    else:
        optimizer = SingleAreaOptimizer(
            data=data,
            pv_params=pv_params,
            ess_params=ess_params,
            ev_params=ev_params,
            ac_params=ac_params,** config
        )

    results = optimizer.solve()

    # 输出结果
    if results:
        print(f"\n{'多区域' if config['use_multi_area'] else '单区域'}优化结果:")
        print(f"总目标函数值: {results['objective_value']:.2f}")
        print(f"电网购电功率: {results['global']['Grid_Power_MW']:.3f} MW")
        print(f"总有功负荷削减: {results['global']['Total_LoadShed_P_MW']:.6f} MW")

        # 可视化
        visualize_results(
            results,
            is_multi_area=config['use_multi_area'],
            control_areas=control_areas if config['use_multi_area'] else None
        )


if __name__ == "__main__":
    main()
