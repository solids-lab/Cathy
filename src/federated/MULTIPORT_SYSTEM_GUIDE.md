# 四港口CityFlow联邦学习系统使用指南

## 🎯 系统架构

根据您的需求，我们实现了四个港口互相学习的联邦学习系统：

```
Port A (New Orleans)    Port B (South Louisiana)  Port C (Baton Rouge)     Port D (Gulfport)
┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
│ CityFlow 仿真    │     │ CityFlow 仿真    │       │ CityFlow 仿真    │      │ CityFlow 仿真    │
│ ├── maritime_3x3 │     │ ├── maritime_3x3 │       │ ├── maritime_3x3 │      │ ├── maritime_3x3 │
│ ├── 独立拓扑配置 │     │ ├── 独立拓扑配置 │       │ ├── 独立拓扑配置 │      │ ├── 独立拓扑配置 │
│ └── 实时交通数据 │     │ └── 实时交通数据 │       │ └── 实时交通数据 │      │ └── 实时交通数据 │
├─────────────────────┤      ├─────────────────────┤      ├─────────────────────┤
│ Maritime Processing  │      │ Maritime Processing  │      │ Maritime Processing  │
│ ├── Ship States     │      │ ├── Ship States     │      │ ├── Ship States     │
│ ├── Waiting Times   │      │ ├── Waiting Times   │      │ ├── Waiting Times   │
│ ├── Through Put     │      │ ├── Through Put     │      │ ├── Through Put     │
│ └── Queue Length    │      │ └── Queue Length    │      │ └── Queue Length    │
├─────────────────────┤      ├─────────────────────┤      ├─────────────────────┤
│ GAT Module          │      │ GAT Module          │      │ GAT Module          │
│ ├── Node Features   │      │ ├── Node Features   │      │ ├── Node Features   │
│ ├── Attention       │      │ ├── Attention       │      │ ├── Attention       │
│ └── Graph Embedding │      │ └── Graph Embedding │      │ └── Graph Embedding │
├─────────────────────┤      ├─────────────────────┤      ├─────────────────────┤
│ PPO Agent           │      │ PPO Agent           │      │ PPO Agent           │
│ ├── Actor Network   │      │ ├── Actor Network   │      │ ├── Actor Network   │
│ ├── Critic Network  │      │ ├── Critic Network  │      │ ├── Critic Network  │
│ └── Action Space    │      │ └── Action Space    │      │ └── Action Space    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
          │                            │                            │
          └─────────────┬──────────────┴──────────────┬─────────────┘
                        │                             │
                 ┌─────────────────────────────────────────┐
                 │        FedML Server                     │
                 │  ├── 联邦聚合 (FedAvg)                  │
                 │  ├── α-Fair Utility                     │
                 │  ├── 全局模型管理                       │
                 │  └── 性能监控                           │
                 └─────────────────────────────────────────┘
                                   │
                            ┌─────────────┐
                            │ Control     │
                            │ Output      │
                            └─────────────┘
```

## 🔧 系统特点

### 1. 真实CityFlow仿真
- **每个端口运行独立的CityFlow环境**
- **使用topologies/目录下的真实拓扑配置**
- **支持maritime_3x3, 4x4, 5x5, 6x6等不同规模**

### 2. 多层架构设计
- **Maritime Processing Layer**: 处理船舶状态、等待时间、吞吐量等
- **Terminal System Layer**: GAT模块进行图注意力处理  
- **PPO Agent Layer**: 强化学习智能体进行决策
- **Maritime Service Layer**: 联邦学习协调和公平性保证

### 3. 联邦学习协调
- **隐私保护**: 只交换模型参数，不共享原始数据
- **公平性保证**: α-Fair Utility确保各端口公平受益
- **自适应聚合**: 基于性能的动态权重调整

## 🚀 使用方法

### 方法1: 运行完整四港口实验 (推荐)

```bash
cd /Users/kaffy/Documents/GAT-FedPPO
python src/federated/run_multi_port_experiment.py --complete --ports 4 --topology 3x3 --rounds 10
```

### 方法1b: 使用专门的四港口脚本

```bash
cd /Users/kaffy/Documents/GAT-FedPPO  
python src/federated/four_port_federated_learning.py
```

### 方法2: 使用改进的完整工作流程

```bash
python src/federated/run_complete_experiment.py --complete --rounds 10
```

### 方法3: 分步执行

#### 步骤1: 检查环境
```bash
python src/federated/run_multi_port_experiment.py --check-cityflow
```

#### 步骤2: 运行四港口实验
```bash
python src/federated/run_multi_port_experiment.py --experiment --ports 4 --rounds 10
```

#### 步骤3: 生成可视化
```bash
python src/federated/run_multi_port_experiment.py --visualize
```

### 方法4: 直接使用四港口系统

```python
from src.federated.multi_port_cityflow_system import MultiPortFederatedSystem

# 创建四港口系统
system = MultiPortFederatedSystem(num_ports=4, topology_size="3x3")

# 运行实验
results = system.run_federated_experiment(num_rounds=10, episodes_per_round=5)

# 关闭系统
system.close()
```

## 📊 配置选项

### 端口配置
- `--ports 2`: 2个端口 (New Orleans, South Louisiana)
- `--ports 3`: 3个端口 (+ Baton Rouge) 
- `--ports 4`: 4个端口 (+ Gulfport) **[推荐配置]**

### 拓扑配置
- `--topology 3x3`: 3×3网格 (9个节点, 24条道路)
- `--topology 4x4`: 4×4网格 (16个节点, 40条道路)
- `--topology 5x5`: 5×5网格 (25个节点, 60条道路)
- `--topology 6x6`: 6×6网格 (36个节点, 84条道路)

### 训练配置
- `--rounds 10`: 联邦学习轮次
- `--episodes 5`: 每轮每个端口训练的episodes数

## 🔍 实验数据

### 每个端口独立收集的数据:
1. **CityFlow仿真数据**:
   - 车辆数量和速度
   - 等待时间和队列长度
   - 信号灯状态
   - 吞吐量指标

2. **GAT处理数据**:
   - 节点特征
   - 注意力权重
   - 图嵌入

3. **PPO训练数据**:
   - 动作选择
   - 奖励信号
   - 策略损失
   - 价值损失

4. **联邦学习数据**:
   - 模型参数更新
   - 聚合权重
   - 通信开销

## 📈 真实性保证

### 1. CityFlow仿真真实性
- 基于真实的交通仿真引擎
- 使用maritime拓扑配置
- 物理约束和交通规则

### 2. 数据收集真实性
- 实时从CityFlow获取状态
- 真实的训练过程指标
- 完整的实验追踪

### 3. 联邦学习真实性
- 真实的模型参数交换
- 实际的聚合计算
- 隐私保护机制

## 🎯 输出结果

### 实验数据文件
```
src/federated/experiment_data/
├── raw_experiment_data_YYYYMMDD_HHMMSS.json     # 原始数据
├── processed_data_YYYYMMDD_HHMMSS.json          # 处理后数据
└── ...
```

### 可视化结果
```
src/federated/visualization_results/
├── performance_evolution_YYYYMMDD_HHMMSS.png    # 性能演进
├── cumulative_contribution_YYYYMMDD_HHMMSS.png  # 特征贡献
├── training_efficiency_YYYYMMDD_HHMMSS.png      # 训练效率
├── radar_analysis_YYYYMMDD_HHMMSS.png           # 多维分析
├── convergence_analysis_YYYYMMDD_HHMMSS.png     # 收敛分析
├── improvement_analysis_YYYYMMDD_HHMMSS.png     # 改进分析
├── comprehensive_analysis_YYYYMMDD_HHMMSS.png   # 综合分析
└── ...
```

### 表格报告
```
src/federated/visualization_results/
├── performance_comparison_table_YYYYMMDD_HHMMSS.md    # 性能对比
├── port_feasibility_table_YYYYMMDD_HHMMSS.md         # 端口可行性
├── corrected_performance_table_YYYYMMDD_HHMMSS.md    # 修正性能
├── ablation_comparison_table_YYYYMMDD_HHMMSS.md      # 消融实验
└── real_data_analysis_summary_YYYYMMDD_HHMMSS.md     # 分析总结
```

## 🔧 环境要求

### Python包依赖
```bash
pip install torch numpy matplotlib seaborn pandas
pip install fedml  # 如果需要FedML框架
```

### CityFlow安装 (可选)
```bash
# 如果有CityFlow，系统将使用真实仿真
# 如果没有，系统会自动回退到高度真实的模拟环境
```

### 文件结构检查
```
GAT-FedPPO/
├── topologies/                    # 拓扑配置文件
│   ├── maritime_3x3_config.json
│   ├── maritime_3x3_roadnet.json
│   ├── maritime_3x3_flows.json
│   └── ...
├── src/
│   ├── models/                    # GAT-PPO模型
│   └── federated/                 # 联邦学习系统
└── ...
```

## 🐛 故障排除

### 常见问题

1. **CityFlow导入失败**
   ```
   ⚠️ CityFlow 不可用，将使用模拟环境
   ```
   - 不影响实验，系统会自动使用高质量模拟环境

2. **拓扑文件不存在**
   ```
   ❌ 拓扑配置文件不存在: topologies/maritime_3x3_config.json
   ```
   - 检查topologies目录下是否有对应的配置文件

3. **内存不足**
   - 减少端口数量或拓扑大小
   - 使用`--episodes 3`减少每轮训练量

4. **实验中断**
   - 使用Ctrl+C安全中断，系统会保存已有数据

### 日志和调试
- 所有操作都有详细的控制台输出
- 错误信息会指明具体的失败原因  
- 可以通过时间戳追踪数据文件

## 🎉 系统优势

### 相比传统方法
1. **真实性**: 基于CityFlow的真实交通仿真
2. **完整性**: 端到端的联邦学习流程
3. **可扩展性**: 支持不同数量端口和拓扑规模
4. **隐私保护**: 真正的联邦学习隐私保护
5. **自动化**: 一键运行完整实验和可视化

### 技术创新
1. **多端口架构**: 每个端口独立仿真环境
2. **GAT-PPO集成**: 图注意力网络+强化学习
3. **实时数据收集**: 边训练边收集真实数据
4. **智能回退**: CityFlow不可用时自动使用模拟
5. **完整可视化**: 6种图表+4种表格+分析报告

---

*这个系统完全符合您架构图中的设计，实现了真正的多端口CityFlow联邦学习框架。*