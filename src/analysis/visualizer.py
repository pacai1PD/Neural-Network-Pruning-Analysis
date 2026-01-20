"""
可视化分析模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, save_dir='results'):
        """
        初始化可视化器
        
        Args:
            save_dir: 结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_pruning_comparison(self, results_df, save_path=None):
        """
        绘制剪枝策略对比图
        
        Args:
            results_df: 结果DataFrame
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 准确率 vs 剪枝率
        ax1 = axes[0, 0]
        for method in results_df['method'].unique():
            if method != 'dense':
                method_data = results_df[results_df['method'] == method]
                ax1.plot(method_data['pruning_rate'], method_data['accuracy'], 
                        marker='o', linewidth=2, markersize=8, label=method.upper())
        
        dense_acc = results_df[results_df['method'] == 'dense']['accuracy'].values[0]
        ax1.axhline(y=dense_acc, color='r', linestyle='--', linewidth=2, label='Dense Baseline')
        ax1.set_xlabel('剪枝率', fontsize=12)
        ax1.set_ylabel('准确率 (%)', fontsize=12)
        ax1.set_title('不同剪枝策略的准确率对比', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 参数量 vs 剪枝率
        ax2 = axes[0, 1]
        for method in results_df['method'].unique():
            if method != 'dense':
                method_data = results_df[results_df['method'] == method]
                ax2.plot(method_data['pruning_rate'], method_data['params'], 
                        marker='s', linewidth=2, markersize=8, label=method.upper())
        
        dense_params = results_df[results_df['method'] == 'dense']['params'].values[0]
        ax2.axhline(y=dense_params, color='r', linestyle='--', linewidth=2, label='Dense Baseline')
        ax2.set_xlabel('剪枝率', fontsize=12)
        ax2.set_ylabel('参数量', fontsize=12)
        ax2.set_title('参数量变化', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 准确率下降率
        ax3 = axes[1, 0]
        dense_acc = results_df[results_df['method'] == 'dense']['accuracy'].values[0]
        for method in results_df['method'].unique():
            if method != 'dense':
                method_data = results_df[results_df['method'] == method]
                accuracy_drop = dense_acc - method_data['accuracy']
                ax3.plot(method_data['pruning_rate'], accuracy_drop, 
                        marker='^', linewidth=2, markersize=8, label=method.upper())
        ax3.set_xlabel('剪枝率', fontsize=12)
        ax3.set_ylabel('准确率下降 (%)', fontsize=12)
        ax3.set_title('准确率下降分析', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. 效率对比（准确率/参数量）
        ax4 = axes[1, 1]
        for method in results_df['method'].unique():
            if method != 'dense':
                method_data = results_df[results_df['method'] == method]
                efficiency = method_data['accuracy'] / method_data['params'] * 1000
                ax4.plot(method_data['pruning_rate'], efficiency, 
                        marker='d', linewidth=2, markersize=8, label=method.upper())
        ax4.set_xlabel('剪枝率', fontsize=12)
        ax4.set_ylabel('效率 (准确率/参数量 × 1000)', fontsize=12)
        ax4.set_title('模型效率对比', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'pruning_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")
        
        plt.close()
    
    def plot_weight_distribution(self, weight_stats, save_path=None):
        """绘制权重分布对比图"""
        if not weight_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        layers = list(weight_stats.keys())
        kept_means = [weight_stats[l]['kept_mean'] for l in layers]
        pruned_means = [weight_stats[l]['pruned_mean'] for l in layers]
        separations = [weight_stats[l]['separation'] for l in layers]
        
        # 1. 保留权重 vs 剪除权重均值对比
        ax1 = axes[0, 0]
        x = np.arange(len(layers))
        width = 0.35
        ax1.bar(x - width/2, kept_means, width, label='保留权重', alpha=0.8)
        ax1.bar(x + width/2, pruned_means, width, label='剪除权重', alpha=0.8)
        ax1.set_xlabel('层名称', fontsize=12)
        ax1.set_ylabel('权重绝对值均值', fontsize=12)
        ax1.set_title('保留权重与剪除权重的均值对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 权重分离度
        ax2 = axes[0, 1]
        ax2.barh(range(len(layers)), separations, alpha=0.8, color='green')
        ax2.set_yticks(range(len(layers)))
        ax2.set_yticklabels([l.split('.')[-1] for l in layers])
        ax2.set_xlabel('权重分离度', fontsize=12)
        ax2.set_title('权重分离度分析（保留-剪除）', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. 保留比例
        ax3 = axes[1, 0]
        kept_ratios = [weight_stats[l]['kept_ratio'] for l in layers]
        ax3.bar(range(len(layers)), kept_ratios, alpha=0.8, color='blue')
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right')
        ax3.set_ylabel('保留比例', fontsize=12)
        ax3.set_title('各层权重保留比例', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 权重数量对比
        ax4 = axes[1, 1]
        kept_counts = [weight_stats[l]['kept_count'] for l in layers]
        pruned_counts = [weight_stats[l]['pruned_count'] for l in layers]
        x = np.arange(len(layers))
        ax4.bar(x - width/2, kept_counts, width, label='保留数量', alpha=0.8)
        ax4.bar(x + width/2, pruned_counts, width, label='剪除数量', alpha=0.8)
        ax4.set_xlabel('层名称', fontsize=12)
        ax4.set_ylabel('权重数量', fontsize=12)
        ax4.set_title('各层权重数量对比', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, 'weight_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"权重分布图已保存至 {save_path}")
        plt.close()
    
    def plot_training_history(self, history, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # 损失曲线
        ax1 = axes[0]
        ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, history['val_losses'], 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('训练和验证损失', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2 = axes[1]
        ax2.plot(epochs, history['train_accuracies'], 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, history['val_accuracies'], 'r-', label='验证准确率', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('训练和验证准确率', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, 'training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"训练历史图已保存至 {save_path}")
        plt.close()
