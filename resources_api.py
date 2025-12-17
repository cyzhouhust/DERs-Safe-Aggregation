import os

# 定义文件路径
FILE_PATH = 'vpp_data.txt'
# 定义列名（对应数据结构）
HEADERS = [
    'id', 'type', 'site', 'rated_capacity', 
    'up_capacity', 'down_capacity', 'resp_time', 'status'
]
# 字段中文映射（用于提示用户输入）
FIELD_PROMPTS = {
    'id': '设备ID (例如: EV-New-01)',
    'type': '种类 (例如: 充电桩)',
    'site': '节点位置 (例如: bus 12)',
    'rated_capacity': '额定容量 (例如: 0.5 MW)',
    'up_capacity': '上行容量 (例如: 200)',
    'down_capacity': '下行容量 (例如: 150)',
    'resp_time': '调节时间 (例如: 5 min)',
    'status': '设备状态 (例如: 可用)'
}
def initialize_dummy_data():
    """初始化测试数据"""
    if not os.path.exists(FILE_PATH):
        initial_data = [
            "AC-Cluster-01,空调,bus 10,1.5 MW,600,900,30 min,可用",
            "EV-Hub-03,充电桩,bus 15,0.8 MW,400,300,5 min,可用",
            "PV-Block-12,光伏,bus 18,3.5 MW,3500,0,即时,可用",
            "BESS-02,储能,bus 20,1.0/2.0 MWh,800,800,1s,可用",
            "AC-Cluster-02,空调,bus 25,2.2 MW,880,1320,30 min,可用",
        ]
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(",".join(HEADERS) + "\n")
            f.write("\n".join(initial_data))

def read_data_from_txt():
    """读取数据并返回列表"""
    data_list = []
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) == len(HEADERS):
                    item = {HEADERS[i]: parts[i] for i in range(len(HEADERS))}
                    data_list.append(item)
    return data_list

def save_to_txt(new_record):
    """追加写入一条数据"""
    # 处理输入中可能包含的英文逗号，防止破坏CSV格式
    values = [str(new_record[k]).replace(',', '，') for k in HEADERS]
    line_str = ",".join(values)
    
    with open(FILE_PATH, 'a', encoding='utf-8') as f:
        f.write("\n" + line_str)
    
    # 写入后立即重新读取，返回最新数据集
    return read_data_from_txt()
def print_table(data):
    """打印漂亮的表格"""
    if not data:
        print("暂无数据")
        return
    print("-" * 110)
    print(f"{'ID':<15} {'种类':<8} {'区域':<12} {'额定容量':<15} {'Up':<6} {'Down':<6} {'时间':<10} {'状态'}")
    print("-" * 110)
    for item in data:
        print(f"{item['id']:<15} {item['type']:<8} {item['site']:<12} {item['rated_capacity']:<15} {item['up_capacity']:<6} {item['down_capacity']:<6} {item['resp_time']:<10} {item['status']}")
    print("-" * 110)
#输出新增的资源后返回的资源列表，新增资源的接口函数，需封装
def terminal_input_mode():
    """
    终端交互录入函数
    """
    print("\n" + "="*40)
    print("      进入新增资源录入模式")
    print("      (按 Ctrl+C 可随时退出)")
    print("="*40 + "\n")
    new_record = {}
    # 遍历字段，逐个询问用户
    for key in HEADERS:
        prompt_text = FIELD_PROMPTS.get(key, key)
        while True:
            # 获取终端输入
            user_input = input(f"请输入 [{prompt_text}]: ").strip()
            # 简单的非空校验
            if user_input:
                new_record[key] = user_input
                break
            else:
                print("⚠️ 输入不能为空，请重新输入。")
    print("\n正在保存数据...")
    # 调用保存逻辑
    updated_data = save_to_txt(new_record)
    print("✅ 数据保存成功！\n")
    return updated_data

# ==========================================
# 主程序逻辑
# ==========================================
if __name__ == '__main__':
    # 1. 确保文件存在
    initialize_dummy_data()
    # 2. 显示当前数据
    print("\n--- 当前数据库内容 ---")
    current_data = read_data_from_txt()
    print_table(current_data) #展示在资源列表
    while True:
        # 3. 询问是否新增
        choice = input("\n是否需要新增一条资源数据？(y/n): ").strip().lower()
        if choice == 'y':
            # 4. 进入终端录入模式 -> 获取输入 -> 写入TXT -> 返回新数据
            latest_dataset = terminal_input_mode()
            # 5. 显示更新后的列表terminal_input_mode()
            print("--- 更新后的完整数据库 ---")
            print_table(latest_dataset)
        elif choice == 'n':
            print("程序已退出。")
            break
        else:
            print("请输入 y 或 n")