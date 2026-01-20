"""
CNN模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单的CNN模型用于CIFAR-10分类
    
    架构：
    - 3个卷积层 (32→64→128通道)
    - 2个全连接层 (256→10)
    - 最大池化和Dropout正则化
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        初始化模型
        
        Args:
            num_classes: 分类类别数
            dropout_rate: Dropout比率
        """
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """前向传播"""
        # 第一组卷积+池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二组卷积+池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三组卷积+池化
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_layer_info(self):
        """获取各层信息"""
        layer_info = []
        for name, param in self.named_parameters():
            layer_info.append({
                'name': name,
                'shape': tuple(param.shape),
                'num_params': param.numel()
            })
        return layer_info


class VGGLike(nn.Module):
    """
    VGG-like模型（用于对比实验）
    """
    
    def __init__(self, num_classes=10, width_factor=1):
        """
        Args:
            num_classes: 分类类别数
            width_factor: 宽度因子（用于控制模型大小）
        """
        super(VGGLike, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, int(64 * width_factor), 3, padding=1),
            nn.BatchNorm2d(int(64 * width_factor)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(64 * width_factor), int(64 * width_factor), 3, padding=1),
            nn.BatchNorm2d(int(64 * width_factor)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(int(64 * width_factor), int(128 * width_factor), 3, padding=1),
            nn.BatchNorm2d(int(128 * width_factor)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width_factor), int(128 * width_factor), 3, padding=1),
            nn.BatchNorm2d(int(128 * width_factor)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(int(128 * width_factor), int(256 * width_factor), 3, padding=1),
            nn.BatchNorm2d(int(256 * width_factor)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(int(256 * width_factor) * 4 * 4, int(512 * width_factor)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(512 * width_factor), num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
