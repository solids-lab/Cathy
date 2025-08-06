#!/bin/bash
"""
启动分布式联邦学习服务器
在终端1运行此脚本
"""

echo "🏢 启动分布式联邦学习服务器"
echo "=========================================="
echo "服务器地址: localhost:8888"
echo "最大轮次: 10"
echo "最少客户端: 2"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# 启动服务器
python src/federated/distributed_federated_server.py \
    --host localhost \
    --port 8888 \
    --min_clients 2 \
    --max_rounds 10

echo "🔒 服务器已停止"