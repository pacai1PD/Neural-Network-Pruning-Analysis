"""
数据加载和预处理模块
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split


class CIFAR10DataLoader:
    """CIFAR-10数据加载器"""
    
    def __init__(self, root='./data', batch_size=128, num_workers=2, 
                 val_split=0.1, random_seed=42):
        """
        初始化数据加载器
        
        Args:
            root: 数据存储路径
            batch_size: 批次大小
            num_workers: 数据加载线程数
            val_split: 验证集比例
            random_seed: 随机种子
        """
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_seed = random_seed
        
        # CIFAR-10数据集的均值和标准差
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
    def get_transforms(self, train=True):
        """获取数据变换"""
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
    
    def load_data(self):
        """加载数据集"""
        # 训练集
        train_transform = self.get_transforms(train=True)
        trainset = torchvision.datasets.CIFAR10(
            root=self.root, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        # 测试集
        test_transform = self.get_transforms(train=False)
        testset = torchvision.datasets.CIFAR10(
            root=self.root, 
            train=False, 
            download=True, 
            transform=test_transform
        )
        
        # 从训练集中划分验证集
        train_indices, val_indices = train_test_split(
            range(len(trainset)),
            test_size=self.val_split,
            random_state=self.random_seed,
            shuffle=True
        )
        
        train_subset = Subset(trainset, train_indices)
        val_subset = Subset(trainset, val_indices)
        
        # 创建数据加载器
        trainloader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        valloader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        testloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # 数据集信息
        dataset_info = {
            'train_size': len(train_subset),
            'val_size': len(val_subset),
            'test_size': len(testset),
            'num_classes': 10,
            'classes': trainset.classes
        }
        
        return trainloader, valloader, testloader, dataset_info
    
    def get_class_names(self):
        """获取类别名称"""
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
