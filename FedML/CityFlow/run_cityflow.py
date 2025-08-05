#!/usr/bin/env python3
"""
调试MASS配置文件的脚本
"""
import json
import os


def check_config_files():
    """检查所有配置文件"""
    print("🔍 检查MASS配置文件...")
    print("=" * 50)

    # 检查config.json
    config_path = "examples/config.json"
    if os.path.exists(config_path):
        print(f"✅ 找到配置文件: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("配置内容:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        return None

    print()

    # 检查roadnet.json
    roadnet_path = os.path.join(config.get('dir', ''), config.get('roadnetFile', ''))
    print(f"📍 检查路网文件: {roadnet_path}")

    if os.path.exists(roadnet_path):
        print("✅ 路网文件存在")
        with open(roadnet_path, 'r') as f:
            roadnet = json.load(f)

        # 提取道路ID
        roads = []
        if 'roads' in roadnet:
            for road in roadnet['roads']:
                roads.append(road['id'])

        print(f"可用道路 ({len(roads)}个):")
        for road in roads:
            print(f"  - {road}")
    else:
        print("❌ 路网文件不存在")
        roads = []

    print()

    # 检查flow.json
    flow_path = os.path.join(config.get('dir', ''), config.get('flowFile', ''))
    print(f"🚢 检查船流文件: {flow_path}")

    if os.path.exists(flow_path):
        print("✅ 船流文件存在")
        with open(flow_path, 'r') as f:
            flows = json.load(f)

        print(f"船流配置 ({len(flows)}个):")
        for i, flow in enumerate(flows):
            route = flow.get('route', [])
            print(f"  船流 {i}: 路线 {route}")

            # 检查路线是否有效
            for road in route:
                if road in roads:
                    print(f"    ✅ {road} - 有效")
                else:
                    print(f"    ❌ {road} - 无效!")
    else:
        print("❌ 船流文件不存在")

    return roads


def create_working_config(available_roads):
    """创建可工作的配置"""
    if not available_roads:
        print("\n⚠️  没有可用道路，无法创建配置")
        return

    print(f"\n🔧 基于可用道路创建新的ship_flow.json:")

    # 创建新的flow配置
    new_flow = []

    for i, road in enumerate(available_roads[:3]):  # 最多使用前3条道路
        flow_config = {
            "vehicle": {
                "length": 50.0 + i * 20,  # 不同大小的船
                "width": 10.0 + i * 2,
                "maxPosAcc": 0.5,
                "maxNegAcc": 1.0,
                "usualPosAcc": 0.3,
                "usualNegAcc": 0.8,
                "minGap": 20.0 + i * 10,
                "maxSpeed": 8.0 - i * 1,
                "headwayTime": 3.0 + i
            },
            "route": [road],
            "interval": 60.0 + i * 30,
            "startTime": 10.0 + i * 20,
            "endTime": 600.0
        }
        new_flow.append(flow_config)

    # 保存新配置
    new_flow_path = "examples/ship_flow_fixed.json"
    with open(new_flow_path, 'w') as f:
        json.dump(new_flow, f, indent=2)

    print(f"✅ 新的船流配置已保存到: {new_flow_path}")
    print("新配置内容:")
    for i, flow in enumerate(new_flow):
        print(f"  船流 {i}: 路线 {flow['route']}, 间隔 {flow['interval']}s")

    # 创建更新的config.json
    config_path = "examples/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 更新flowFile指向新文件
        config['flowFile'] = 'ship_flow_fixed.json'

        new_config_path = "examples/config_fixed.json"
        with open(new_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 更新的配置文件已保存到: {new_config_path}")
        print("使用方法:")
        print(f"  将你的脚本中的配置路径改为: '{new_config_path}'")


def main():
    print("🌊 MASS配置文件调试工具")
    print("=" * 60)

    # 检查当前配置
    roads = check_config_files()

    # 如果有可用道路，创建修复版本
    if roads:
        create_working_config(roads)
    else:
        print("\n💡 建议:")
        print("1. 确保examples/目录存在")
        print("2. 将修复后的roadnet.json放入examples/目录")
        print("3. 将修复后的flow.json放入examples/目录")
        print("4. 确保config.json指向正确的文件")


if __name__ == "__main__":
    main()