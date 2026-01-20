"""
深度分析模块
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class PruningAnalyzer:
    """剪枝深度分析器"""
    
    def __init__(self, results_df):
        """
        初始化分析器
        
        Args:
            results_df: 结果DataFrame
        """
        self.results_df = results_df
    
    def analyze_pruning_rate_impact(self):
        """分析剪枝率对性能的影响"""
        analysis = {}
        
        dense_acc = self.results_df[self.results_df['method'] == 'dense']['accuracy'].values[0]
        
        for method in self.results_df['method'].unique():
            if method == 'dense':
                continue
            
            method_data = self.results_df[self.results_df['method'] == method].copy()
            method_data['accuracy_drop'] = dense_acc - method_data['accuracy']
            method_data['relative_drop'] = method_data['accuracy_drop'] / dense_acc * 100
            
            # 计算不同剪枝率区间的性能
            low_pruning = method_data[method_data['pruning_rate'] <= 0.3]
            mid_pruning = method_data[(method_data['pruning_rate'] > 0.3) & 
                                     (method_data['pruning_rate'] <= 0.7)]
            high_pruning = method_data[method_data['pruning_rate'] > 0.7]
            
            analysis[method] = {
                'low_pruning_avg_drop': low_pruning['accuracy_drop'].mean() if len(low_pruning) > 0 else 0,
                'mid_pruning_avg_drop': mid_pruning['accuracy_drop'].mean() if len(mid_pruning) > 0 else 0,
                'high_pruning_avg_drop': high_pruning['accuracy_drop'].mean() if len(high_pruning) > 0 else 0,
                'best_pruning_rate': method_data.loc[method_data['accuracy'].idxmax(), 'pruning_rate'],
                'best_accuracy': method_data['accuracy'].max(),
                'worst_pruning_rate': method_data.loc[method_data['accuracy'].idxmin(), 'pruning_rate'],
                'worst_accuracy': method_data['accuracy'].min()
            }
        
        return analysis
    
    def find_optimal_pruning_rate(self, target_accuracy_drop=5.0):
        """找到最优剪枝率（在目标准确率下降范围内）"""
        optimal_rates = {}
        
        dense_acc = self.results_df[self.results_df['method'] == 'dense']['accuracy'].values[0]
        target_acc = dense_acc - target_accuracy_drop
        
        for method in self.results_df['method'].unique():
            if method == 'dense':
                continue
            
            method_data = self.results_df[self.results_df['method'] == method].copy()
            # 找到准确率最接近目标且不低于目标的剪枝率
            valid_data = method_data[method_data['accuracy'] >= target_acc]
            
            if len(valid_data) > 0:
                # 选择准确率最高的
                optimal = valid_data.loc[valid_data['accuracy'].idxmax()]
                optimal_rates[method] = {
                    'pruning_rate': optimal['pruning_rate'],
                    'accuracy': optimal['accuracy'],
                    'accuracy_drop': dense_acc - optimal['accuracy'],
                    'params': optimal['params'],
                    'sparsity': optimal['sparsity']
                }
            else:
                optimal_rates[method] = None
        
        return optimal_rates
    
    def compare_methods_at_same_sparsity(self, target_sparsity=0.5):
        """在相同稀疏度下对比不同方法"""
        comparison = {}
        
        for method in self.results_df['method'].unique():
            if method == 'dense':
                continue
            
            method_data = self.results_df[self.results_df['method'] == method].copy()
            # 找到最接近目标稀疏度的数据点
            method_data['sparsity_diff'] = abs(method_data['sparsity'] - target_sparsity)
            closest = method_data.loc[method_data['sparsity_diff'].idxmin()]
            
            comparison[method] = {
                'accuracy': closest['accuracy'],
                'pruning_rate': closest['pruning_rate'],
                'sparsity': closest['sparsity'],
                'params': closest['params']
            }
        
        return comparison
    
    def generate_summary_statistics(self):
        """生成汇总统计"""
        stats = {}
        
        dense_row = self.results_df[self.results_df['method'] == 'dense'].iloc[0]
        dense_acc = dense_row['accuracy']
        dense_params = dense_row['params']
        
        stats['dense'] = {
            'accuracy': dense_acc,
            'params': dense_params
        }
        
        for method in self.results_df['method'].unique():
            if method == 'dense':
                continue
            
            method_data = self.results_df[self.results_df['method'] == method]
            
            stats[method] = {
                'max_accuracy': method_data['accuracy'].max(),
                'min_accuracy': method_data['accuracy'].min(),
                'avg_accuracy': method_data['accuracy'].mean(),
                'max_accuracy_drop': dense_acc - method_data['accuracy'].min(),
                'min_accuracy_drop': dense_acc - method_data['accuracy'].max(),
                'avg_accuracy_drop': dense_acc - method_data['accuracy'].mean(),
                'best_pruning_rate': method_data.loc[method_data['accuracy'].idxmax(), 'pruning_rate'],
                'compression_ratio_at_best': method_data.loc[method_data['accuracy'].idxmax(), 'params'] / dense_params
            }
        
        return stats
