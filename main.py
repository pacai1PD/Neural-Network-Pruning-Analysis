"""
神经网络剪枝策略效果分析与优化 - 主程序
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 导入自定义模块
from src.utils.data_loader import CIFAR10DataLoader
from src.utils.metrics import MetricsCalculator
from src.models.cnn import SimpleCNN
from src.pruning.pruning_strategies import PruningStrategy
from src.pruning.trainer import ModelTrainer
from src.analysis.visualizer import ResultVisualizer
from src.analysis.analyzer import PruningAnalyzer


def set_random_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_baseline_model(train_loader, val_loader, test_loader, device, epochs=10):
    """训练基线模型"""
    print("\n" + "="*60)
    print("训练完整模型（Dense Baseline）")
    print("="*60)
    
    model = SimpleCNN(num_classes=10)
    trainer = ModelTrainer(model, device, lr=0.001)
    
    history = trainer.train(train_loader, val_loader, epochs=epochs, verbose=True)
    accuracy, preds, labels = trainer.evaluate(test_loader)
    
    params = MetricsCalculator.count_parameters(model)
    model_size = MetricsCalculator.model_size_mb(model)
    
    print(f"\n测试集准确率: {accuracy:.2f}%")
    print(f"模型参数量: {params:,}")
    print(f"模型大小: {model_size:.2f} MB")
    
    return model, trainer, {
        'accuracy': accuracy,
        'params': params,
        'model_size_mb': model_size,
        'history': history
    }


def evaluate_pruning_strategy(model, train_loader, val_loader, test_loader, 
                               device, pruning_rate, method, epochs_finetune=5):
    """评估单个剪枝策略"""
    print(f"\n评估 {method.upper()} 剪枝策略，剪枝率: {pruning_rate:.1%}")
    
    # 创建模型副本
    pruned_model = SimpleCNN(num_classes=10)
    pruned_model.load_state_dict(model.state_dict())
    pruned_model.to(device)
    
    # 创建剪枝策略
    pruner = PruningStrategy(pruned_model)
    
    # 生成剪枝掩码
    masks, sparsity = pruner.get_pruning_mask(pruning_rate, method=method)
    
    # 应用剪枝
    pruner.apply_pruning(masks)
    
    # 统计参数量
    params = MetricsCalculator.count_parameters(pruned_model)
    
    # 微调模型
    trainer = ModelTrainer(pruned_model, device, lr=0.0001)
    trainer.train(train_loader, val_loader, epochs=epochs_finetune, verbose=False)
    
    # 评估性能
    accuracy, preds, labels = trainer.evaluate(test_loader)
    
    # 分析权重分布
    weight_stats = pruner.analyze_weight_distribution(masks)
    
    print(f"  准确率: {accuracy:.2f}%, 参数量: {params:,}, 稀疏度: {sparsity:.2%}")
    
    return {
        'method': method,
        'pruning_rate': pruning_rate,
        'accuracy': accuracy,
        'params': params,
        'sparsity': sparsity,
        'weight_stats': weight_stats
    }


def run_full_experiment(config):
    """运行完整实验"""
    # 设置随机种子
    set_random_seed(config['random_seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载CIFAR-10数据集...")
    data_loader = CIFAR10DataLoader(
        root=config['data_root'],
        batch_size=config['batch_size'],
        val_split=config['val_split']
    )
    train_loader, val_loader, test_loader, dataset_info = data_loader.load_data()
    print(f"训练集: {dataset_info['train_size']} 样本")
    print(f"验证集: {dataset_info['val_size']} 样本")
    print(f"测试集: {dataset_info['test_size']} 样本")
    
    # 训练基线模型
    baseline_model, baseline_trainer, baseline_results = train_baseline_model(
        train_loader, val_loader, test_loader, device, epochs=config['baseline_epochs']
    )
    
    # 保存基线结果
    results = [{
        'method': 'dense',
        'pruning_rate': 0.0,
        'accuracy': baseline_results['accuracy'],
        'params': baseline_results['params'],
        'sparsity': 0.0,
        'model_size_mb': baseline_results['model_size_mb']
    }]
    
    # 评估不同剪枝策略
    print("\n" + "="*60)
    print("评估不同剪枝策略")
    print("="*60)
    
    methods = config['pruning_methods']
    pruning_rates = config['pruning_rates']
    
    for method in methods:
        for pruning_rate in pruning_rates:
            result = evaluate_pruning_strategy(
                baseline_model, train_loader, val_loader, test_loader,
                device, pruning_rate, method, epochs_finetune=config['finetune_epochs']
            )
            results.append(result)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    os.makedirs(config['results_dir'], exist_ok=True)
    results_file = os.path.join(config['results_dir'], 'pruning_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存至 {results_file}")
    
    # 保存CSV格式
    csv_file = os.path.join(config['results_dir'], 'pruning_results.csv')
    results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"结果CSV已保存至 {csv_file}")
    
    # 可视化结果
    print("\n生成可视化图表...")
    visualizer = ResultVisualizer(save_dir=config['results_dir'])
    visualizer.plot_pruning_comparison(
        results_df, 
        save_path=os.path.join(config['results_dir'], 'pruning_comparison.png')
    )
    
    # 分析权重分布（使用L1剪枝50%的结果）
    print("\n分析权重分布...")
    l1_50_result = next((r for r in results if r['method'] == 'l1' and abs(r['pruning_rate'] - 0.5) < 0.01), None)
    if l1_50_result and 'weight_stats' in l1_50_result:
        visualizer.plot_weight_distribution(
            l1_50_result['weight_stats'],
            save_path=os.path.join(config['results_dir'], 'weight_distribution.png')
        )
    
    # 深度分析
    print("\n进行深度分析...")
    analyzer = PruningAnalyzer(results_df)
    
    # 剪枝率影响分析
    pruning_impact = analyzer.analyze_pruning_rate_impact()
    
    # 最优剪枝率分析
    optimal_rates = analyzer.find_optimal_pruning_rate(target_accuracy_drop=5.0)
    
    # 相同稀疏度对比
    same_sparsity_comparison = analyzer.compare_methods_at_same_sparsity(target_sparsity=0.5)
    
    # 汇总统计
    summary_stats = analyzer.generate_summary_statistics()
    
    # 保存分析结果
    analysis_results = {
        'pruning_impact': pruning_impact,
        'optimal_rates': optimal_rates,
        'same_sparsity_comparison': same_sparsity_comparison,
        'summary_stats': summary_stats
    }
    
    analysis_file = os.path.join(config['results_dir'], 'analysis_results.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"分析结果已保存至 {analysis_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    print("\n完整模型性能:")
    print(f"  准确率: {baseline_results['accuracy']:.2f}%")
    print(f"  参数量: {baseline_results['params']:,}")
    
    print("\n各剪枝策略最佳性能:")
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        best = method_data.loc[method_data['accuracy'].idxmax()]
        print(f"  {method.upper()}: 准确率 {best['accuracy']:.2f}% "
              f"(剪枝率 {best['pruning_rate']:.1%}, 参数量 {best['params']:,})")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    
    return results_df, analysis_results


def main():
    """主函数"""
    # 配置参数
    config = {
        'random_seed': 42,
        'data_root': './data',
        'batch_size': 128,
        'val_split': 0.1,
        'baseline_epochs': 10,
        'finetune_epochs': 5,
        'pruning_methods': ['l1', 'l2', 'random', 'lottery'],
        'pruning_rates': [0.1, 0.3, 0.5, 0.7, 0.9],
        'results_dir': 'results'
    }
    
    print("="*60)
    print("神经网络剪枝策略效果分析与优化")
    print("="*60)
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 运行实验
    results_df, analysis_results = run_full_experiment(config)
    
    return results_df, analysis_results


if __name__ == '__main__':
    results_df, analysis_results = main()
