# 🚢 海事GAT-FedPPO联邦学习启动指南

## 📋 系统概述

这是一个**真正的分布式联邦学习系统**，将启动5个独立进程：
- 1个服务器进程（聚合器）
- 4个客户端进程（代表4个真实港口）

## 🗺️ 港口映射关系

| Rank | 港口名称 | 英文标识 | 位置坐标 | 港口类型 |
|------|----------|----------|----------|----------|
| 1 | 新奥尔良港 | new_orleans | 29.9311°N, 90.0831°W | 集装箱港 |
| 2 | 南路易斯安那港 | south_louisiana | 29.7755°N, 90.6420°W | 散货港 |
| 3 | 巴吞鲁日港 | baton_rouge | 30.4515°N, 91.1871°W | 内河港 |
| 4 | 海湾港 | gulfport | 30.3674°N, 89.0928°W | 综合港 |

## 🚀 启动步骤

### 前提条件
```bash
# 确保在项目根目录
cd /Users/kaffy/Documents/GAT-FedPPO

# 检查必要文件存在
ls src/federated/config/maritime_fedml_config.yaml
ls src/federated/run_server.sh
ls src/federated/run_client.sh
```

### 步骤1: 启动联邦服务器
**在终端1中运行：**
```bash
cd src/federated
./run_server.sh
```

**预期输出：**
```
🚢 启动海事GAT-PPO联邦学习服务器
=================================
📋 配置文件: config/maritime_fedml_config.yaml
🔧 启动服务器...
INFO - 🚢 启动海事GAT-PPO联邦学习服务器
INFO - ✅ 配置加载完成
INFO - ✅ 模型创建完成
INFO - ✅ 聚合器创建完成
INFO - 🚀 启动联邦学习服务器...
[FedML] Server waiting for clients...
```

### 步骤2: 启动4个港口客户端

**在终端2中运行（新奥尔良港）：**
```bash
cd src/federated
./run_client.sh 1
```

**在终端3中运行（南路易斯安那港）：**
```bash
cd src/federated
./run_client.sh 2
```

**在终端4中运行（巴吞鲁日港）：**
```bash
cd src/federated
./run_client.sh 3
```

**在终端5中运行（海湾港）：**
```bash
cd src/federated
./run_client.sh 4
```

**每个客户端的预期输出：**
```
🚢 启动海事GAT-PPO联邦学习客户端
=================================
🆔 客户端Rank: 1
🔖 运行ID: maritime_test_20250802_150000
📍 节点名称: new_orleans
📋 配置文件: config/maritime_fedml_config.yaml
🔧 启动客户端...
INFO - 🚢 启动海事GAT-PPO联邦学习客户端: new_orleans
INFO - ✅ 配置加载完成
INFO - ✅ 模型创建完成
INFO - ✅ 训练器创建完成
INFO - 🚀 启动联邦学习客户端 new_orleans...
[FedML] Client new_orleans connected to server...
```

## 🔄 联邦训练流程

### 自动化训练（50轮）
一旦所有4个客户端连接到服务器，系统将自动开始50轮联邦训练：

```
轮次 1/50:
├── 📍 new_orleans: 本地训练 5 episodes → 上传模型参数
├── 📍 south_louisiana: 本地训练 5 episodes → 上传模型参数  
├── 📍 baton_rouge: 本地训练 5 episodes → 上传模型参数
└── 📍 gulfport: 本地训练 5 episodes → 上传模型参数

🔄 服务器聚合: 计算智能权重 → 生成全局模型 → 分发给客户端

轮次 2/50:
├── 📍 各港口使用新的全局模型继续训练...
└── ...

...

轮次 50/50: 训练完成！
```

### 监控训练进度
在服务器终端中，您会看到：
```
Round 1: Aggregating models from 4 clients
├── new_orleans: weight=0.28, avg_reward=750.5
├── south_louisiana: weight=0.26, avg_reward=720.3
├── baton_rouge: weight=0.22, avg_reward=680.1
└── gulfport: weight=0.24, avg_reward=700.8

Global model updated, broadcasting to clients...

Round 2: Aggregating models from 4 clients
├── new_orleans: weight=0.27, avg_reward=775.2 (+24.7)
├── south_louisiana: weight=0.25, avg_reward=745.6 (+25.3)
├── baton_rouge: weight=0.24, avg_reward=695.4 (+15.3)
└── gulfport: weight=0.24, avg_reward=718.9 (+18.1)
```

## 📊 训练结果

训练完成后，您将得到：
1. **全局模型**: `model_cache/global_maritime_model.pt`
2. **训练日志**: `logs/fedml.log`
3. **性能报告**: 各港口的性能提升统计

## 🛑 停止训练

要停止训练，在任意终端按 `Ctrl+C`，系统会优雅关闭。

## 🔧 故障排除

### 常见问题
1. **端口占用**: 如果8888端口被占用，修改配置文件中的端口号
2. **客户端连接失败**: 确保服务器先启动，再启动客户端
3. **Python路径错误**: 确保PYTHONPATH正确设置

### 调试模式
启用详细日志：
```bash
export FEDML_LOG_LEVEL=DEBUG
./run_server.sh
```

---

## ✅ 验证系统是真正的分布式

您可以通过以下方式验证：
1. **进程查看**: `ps aux | grep maritime` - 应该看到5个独立进程
2. **网络连接**: `netstat -an | grep 8888` - 应该看到网络连接
3. **独立故障**: 关闭一个客户端，其他客户端继续运行
4. **跨机部署**: 可以将客户端部署到不同的机器上

🎉 **恭喜！您现在拥有了一个真正的分布式海事联邦学习系统！**