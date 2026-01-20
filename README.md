# 神经网络剪枝策略效果分析与优化项目

## 项目简介

本项目基于**彩票假设**理论，系统性地分析多种神经网络剪枝策略（L1范数、L2范数、随机剪枝、彩票假设剪枝）在CIFAR-10数据集上的效果。项目从**数据层面**、**方法层面**、**分析层面**进行全面处理、建模、分析和验证，输出完整的剪枝策略效果评估报告。

## 项目特点

- ✅ **完整的项目结构**：模块化设计，代码组织清晰
- ✅ **多种剪枝策略**：实现L1、L2、随机、彩票假设四种剪枝方法
- ✅ **系统化数据分析**：多维度指标采集和深度分析
- ✅ **可视化分析**：自动生成对比图表和权重分布图
- ✅ **完整实验报告**：包含数据处理、模型训练、结果分析的完整报告

## 项目结构

```
ML/
├── main.py                                    # 主程序入口
├── config.yaml                                # 配置文件
├── requirements.txt                            # Python依赖包
├── README.md                                  # 项目说明文档
│
├── src/                                       # 源代码目录
│   ├── __init__.py
│   ├── models/                                # 模型定义
│   │   ├── __init__.py
│   │   └── cnn.py                            # CNN模型定义
│   │
│   ├── utils/                                 # 工具函数
│   │   ├── __init__.py
│   │   ├── data_loader.py                    # 数据加载和预处理
│   │   └── metrics.py                        # 评估指标计算
│   │
│   ├── pruning/                               # 剪枝策略
│   │   ├── __init__.py
│   │   ├── pruning_strategies.py             # 剪枝策略实现
│   │   └── trainer.py                        # 模型训练器
│   │
│   └── analysis/                              # 分析和可视化
│       ├── __init__.py
│       ├── visualizer.py                     # 结果可视化
│       └── analyzer.py                        # 深度分析
│
├── data/                                      # 数据目录（自动生成）
│   └── cifar-10-batches-py/                  # CIFAR-10数据集
│
└── results/                                   # 结果目录（自动生成）
    ├── pruning_results.json                   # 实验结果JSON
    ├── pruning_results.csv                    # 实验结果CSV
    ├── analysis_results.json                  # 分析结果JSON
    ├── pruning_comparison.png                 # 剪枝策略对比图
    └── weight_distribution.png                # 权重分布图
```

## 环境要求

- Python 3.7+
- PyTorch 1.12+
- CUDA（可选，用于GPU加速）

## 使用方法

### 基本使用

直接运行主程序：

```bash
python main.py
```

程序将自动：
1. 下载CIFAR-10数据集（首次运行）
2. 训练完整模型（Dense Baseline）
3. 评估不同剪枝策略和剪枝率
4. 生成可视化图表和分析报告

### 配置参数

可以通过修改 `config.yaml` 或直接修改 `main.py` 中的配置来调整实验参数：

- `baseline_epochs`: 基线模型训练轮数（默认10）
- `finetune_epochs`: 剪枝后微调轮数（默认5）
- `pruning_methods`: 剪枝方法列表（默认：['l1', 'l2', 'random', 'lottery']）
- `pruning_rates`: 剪枝率列表（默认：[0.1, 0.3, 0.5, 0.7, 0.9]）

### 输出结果

运行完成后，结果将保存在 `results/` 目录：

- **pruning_results.json**: 完整的实验结果数据（JSON格式）
- **pruning_results.csv**: 实验结果表格（CSV格式）
- **analysis_results.json**: 深度分析结果（剪枝率影响、最优剪枝率等）
- **pruning_comparison.png**: 剪枝策略对比可视化图表
- **weight_distribution.png**: 权重分布分析图表

## 项目模块说明

### 1. 数据层（src/utils/data_loader.py）

- **CIFAR10DataLoader**: 数据加载和预处理
  - 自动下载CIFAR-10数据集
  - 数据增强（随机裁剪、水平翻转）
  - 训练集/验证集/测试集划分
  - 数据归一化

### 2. 方法层

#### 模型定义（src/models/cnn.py）
- **SimpleCNN**: 简单CNN架构（3个卷积层 + 2个全连接层）
- **VGGLike**: VGG-like架构（用于对比实验）

#### 剪枝策略（src/pruning/pruning_strategies.py）
- **L1范数剪枝**: 按权重绝对值大小剪枝
- **L2范数剪枝**: 按权重平方和剪枝
- **随机剪枝**: 随机选择权重剪枝
- **彩票假设剪枝**: 基于初始权重的剪枝（验证彩票假设）

#### 训练器（src/pruning/trainer.py）
- **ModelTrainer**: 模型训练和评估
  - 支持训练和验证
  - 学习率调度
  - 最佳模型保存

### 3. 分析层

#### 可视化（src/analysis/visualizer.py）
- **ResultVisualizer**: 结果可视化
  - 剪枝策略对比图（准确率、参数量、性能下降、效率）
  - 权重分布分析图
  - 训练历史曲线

#### 深度分析（src/analysis/analyzer.py）
- **PruningAnalyzer**: 深度分析器
  - 剪枝率影响分析
  - 最优剪枝率查找
  - 相同稀疏度下方法对比
  - 汇总统计

## 与作业1的关联

本项目与作业1中的"彩票假设"和"迭代权重回收"理论紧密相关：

- ✅ 验证了随机初始化网络中蕴含高性能子网络的假设
- ✅ 实现了基于初始权重的彩票假设剪枝方法
- ✅ 在高剪枝率场景下，可结合"迭代权重回收"算法进一步优化
- ✅ 为剪枝策略选择提供了数据驱动的决策依据

## 注意事项

1. **运行时间**：完整实验可能需要较长时间
   - CPU: 约2-4小时
   - GPU: 约30-60分钟
   - 可通过减少epochs或剪枝率数量快速测试

2. **内存要求**：建议至少4GB可用内存

3. **数据集**：CIFAR-10数据集约170MB，首次运行会自动下载

4. **结果文件**：所有结果保存在 `results/` 目录