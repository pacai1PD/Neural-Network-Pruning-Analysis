"""
剪枝策略实现模块
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class PruningStrategy:
    """剪枝策略基类"""
    
    def __init__(self, model):
        """
        初始化剪枝策略
        
        Args:
            model: 要剪枝的模型
        """
        self.model = model
        self.initial_weights = {}
        self.save_initial_weights()
    
    def save_initial_weights(self):
        """保存初始权重（用于彩票假设）"""
        for name, param in self.model.named_parameters():
            if len(param.shape) > 1:  # 只保存权重矩阵
                self.initial_weights[name] = param.data.clone()
    
    def get_pruning_mask(self, pruning_rate, method='l1'):
        """
        生成剪枝掩码
        
        Args:
            pruning_rate: 剪枝率 (0-1)
            method: 剪枝方法 ('l1', 'l2', 'random', 'lottery')
        
        Returns:
            masks: 掩码字典
            sparsity: 实际稀疏度
        """
        masks = OrderedDict()
        total_params = 0
        pruned_params = 0
        
        for name, param in self.model.named_parameters():
            if len(param.shape) > 1:  # 只剪枝权重矩阵，不剪枝偏置
                if method == 'l1':
                    mask = self._l1_pruning(param, pruning_rate)
                elif method == 'l2':
                    mask = self._l2_pruning(param, pruning_rate)
                elif method == 'random':
                    mask = self._random_pruning(param, pruning_rate)
                elif method == 'lottery':
                    mask = self._lottery_pruning(name, param, pruning_rate)
                else:
                    raise ValueError(f"Unknown pruning method: {method}")
                
                masks[name] = mask
                total_params += param.numel()
                pruned_params += (mask == 0).sum().item()
            else:
                # 偏置项不剪枝
                masks[name] = torch.ones_like(param.data)
        
        sparsity = pruned_params / total_params if total_params > 0 else 0.0
        return masks, sparsity
    
    def _l1_pruning(self, param, pruning_rate):
        """L1范数剪枝：按绝对值大小排序"""
        flat_params = param.data.abs().flatten()
        num_prune = int(pruning_rate * flat_params.numel())
        
        if num_prune >= flat_params.numel():
            return torch.zeros_like(param.data)
        
        threshold_idx = flat_params.numel() - num_prune
        threshold = torch.kthvalue(flat_params, threshold_idx)[0]
        mask = (param.data.abs() > threshold).float()
        
        return mask
    
    def _l2_pruning(self, param, pruning_rate):
        """L2范数剪枝：按平方和排序"""
        flat_params_sq = (param.data ** 2).flatten()
        num_prune = int(pruning_rate * flat_params_sq.numel())
        
        if num_prune >= flat_params_sq.numel():
            return torch.zeros_like(param.data)
        
        threshold_idx = flat_params_sq.numel() - num_prune
        threshold = torch.kthvalue(flat_params_sq, threshold_idx)[0]
        mask = ((param.data ** 2) > threshold).float()
        
        return mask
    
    def _random_pruning(self, param, pruning_rate):
        """随机剪枝"""
        mask = torch.ones_like(param.data)
        num_prune = int(pruning_rate * param.numel())
        
        if num_prune >= param.numel():
            return torch.zeros_like(param.data)
        
        indices = torch.randperm(param.numel())[:num_prune]
        mask_flat = mask.flatten()
        mask_flat[indices] = 0
        mask = mask_flat.view(param.shape)
        
        return mask
    
    def _lottery_pruning(self, name, param, pruning_rate):
        """彩票假设剪枝：基于初始权重的绝对值"""
        if name not in self.initial_weights:
            # 如果没有初始权重，使用当前权重
            initial_weights = param.data.clone()
        else:
            initial_weights = self.initial_weights[name]
        
        flat_initial = initial_weights.abs().flatten()
        num_prune = int(pruning_rate * flat_initial.numel())
        
        if num_prune >= flat_initial.numel():
            return torch.zeros_like(param.data)
        
        threshold_idx = flat_initial.numel() - num_prune
        threshold = torch.kthvalue(flat_initial, threshold_idx)[0]
        mask = (initial_weights.abs() > threshold).float()
        
        return mask
    
    def apply_pruning(self, masks):
        """应用剪枝掩码"""
        for name, param in self.model.named_parameters():
            if name in masks:
                param.data *= masks[name]
    
    def analyze_weight_distribution(self, masks):
        """分析权重分布"""
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if name in masks and len(param.shape) > 1:
                weights = param.data.cpu().numpy().flatten()
                mask = masks[name].cpu().numpy().flatten()
                
                kept_weights = weights[mask == 1]
                pruned_weights = weights[mask == 0]
                
                if len(kept_weights) > 0 and len(pruned_weights) > 0:
                    weight_stats[name] = {
                        'kept_mean': float(np.mean(np.abs(kept_weights))),
                        'kept_std': float(np.std(np.abs(kept_weights))),
                        'kept_median': float(np.median(np.abs(kept_weights))),
                        'pruned_mean': float(np.mean(np.abs(pruned_weights))),
                        'pruned_std': float(np.std(np.abs(pruned_weights))),
                        'pruned_median': float(np.median(np.abs(pruned_weights))),
                        'separation': float(np.mean(np.abs(kept_weights)) - np.mean(np.abs(pruned_weights))),
                        'kept_count': int(len(kept_weights)),
                        'pruned_count': int(len(pruned_weights)),
                        'kept_ratio': float(len(kept_weights) / len(weights))
                    }
        
        return weight_stats
