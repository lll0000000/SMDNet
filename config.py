# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 09:19:48 2025

@author: pc
"""

import torch
import os

class Config:
    """项目配置参数"""
    
    # 数据参数
    max_nodes = 75
    atom_feature_dim = 43
    fps_dim = 2048
    embedding_dim = 128
    abeta_dim = 20
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.0001
    epochs = 100
    dropout = 0.1
    head = 3
    
    # 模型参数
    n_output = 1
    output_dim = 128
    
    # 路径参数
    data_dir = 'data'
    model_dir = 'models'
    output_dir = 'output'
    
    # 文件路径
    train_file = 'data_train.csv'
    val_file = 'data_val.csv'
    test_file = 'data_test.csv'
    model_file = 'SMDNet.pth'
    
    # Aβ蛋白序列
    abeta_sequence = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    seed = 42
    
    # 筛选参数
    mc_dropout_samples = 10
    candidate_percent = 0.4
    num_samples = 100
    num_bins = 5
    mi_weight = 0.5

# 创建必要的目录
def create_directories():
    """创建项目所需的目录"""
    directories = [Config.data_dir, Config.model_dir, Config.output_dir, 'explain']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 初始化目录
create_directories()