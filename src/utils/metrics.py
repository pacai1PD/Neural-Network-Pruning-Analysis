"""
评估指标模块
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def accuracy(outputs, labels):
        """计算准确率"""
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return 100.0 * correct / total
    
    @staticmethod
    def top_k_accuracy(outputs, labels, k=5):
        """计算Top-K准确率"""
        _, top_k_pred = torch.topk(outputs, k, dim=1)
        labels_expanded = labels.view(-1, 1).expand_as(top_k_pred)
        correct = (top_k_pred == labels_expanded).any(dim=1).sum().item()
        total = labels.size(0)
        return 100.0 * correct / total
    
    @staticmethod
    def count_parameters(model):
        """统计模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def count_sparsity(model, mask_dict=None):
        """计算模型稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if mask_dict and name in mask_dict:
                mask = mask_dict[name]
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
            else:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0.0
        return sparsity, total_params, zero_params
    
    @staticmethod
    def model_size_mb(model):
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def detailed_classification_report(model, dataloader, device, class_names=None):
        """生成详细的分类报告"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
