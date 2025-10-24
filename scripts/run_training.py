#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script entry point for SMDNet model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from config import Config
from src.model import CrossAttGATNet_AdaLN
from src.train import train_and_val, test, show_loss_curve, save_result
from src.utils import generate_smile_graph, get_torch_data, set_seed, compute_aac
import pandas as pd

def main():
    """Main training function"""
    print("Starting SMDNet model training...")
    
    # Set random seed for reproducibility
    set_seed(Config.seed)
    
    # Initialize model
    print("Initializing model...")
    model = CrossAttGATNet_AdaLN(
        max_nodes=Config.max_nodes,
        atom_feature_dim=Config.atom_feature_dim, 
        embedding_dim=Config.embedding_dim,
        head=Config.head,
        fps_dim=Config.fps_dim,
        abeta_dim=Config.abeta_dim,
        dropout=Config.dropout,
        n_output=Config.n_output,
        output_dim=Config.output_dim
    )
    model = model.to(Config.device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Compute Aβ protein features
    print("Computing Aβ protein features...")
    abeta_feature = compute_aac(Config.abeta_sequence)
    abeta_feature = torch.tensor(abeta_feature).float().to(Config.device)
    
    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv(os.path.join(Config.data_dir, Config.train_file))
    train_smi = train_data.iloc[:, 1]
    train_y = train_data.iloc[:, 2]
    
    train_graph = generate_smile_graph(train_smi, Config.max_nodes)
    train_loader = get_torch_data(train_smi, train_y, train_graph, Config.batch_size, True)
    
    # Load validation data
    val_data = pd.read_csv(os.path.join(Config.data_dir, Config.val_file))
    val_smi = val_data.iloc[:, 1]
    val_y = val_data.iloc[:, 2]
    
    val_graph = generate_smile_graph(val_smi, Config.max_nodes)
    val_loader = get_torch_data(val_smi, val_y, val_graph, Config.batch_size, False)
    
    # Start training
    print("Starting model training...")
    train_loss_lt, val_loss_lt, train_r2_lt, val_r2_lt, train_rmse_lt, val_rmse_lt, train_mae_lt, val_mae_lt = train_and_val(
        model, train_loader, val_loader, abeta_feature, Config.device, 
        Config.epochs, optimizer, loss_fn, 
        os.path.join(Config.model_dir, Config.model_file)
    )
    
    # Save training results
    save_result(Config.epochs, train_loss_lt, val_loss_lt, train_r2_lt, val_r2_lt, 
                train_rmse_lt, val_rmse_lt, train_mae_lt, val_mae_lt)
    
    # Display loss curves
    show_loss_curve(Config.epochs, train_loss_lt, val_loss_lt)
    
    # Test model performance
    print("Evaluating model on test set...")
    test_data = pd.read_csv(os.path.join(Config.data_dir, Config.test_file))
    test_smi = test_data.iloc[:, 1]
    test_y = test_data.iloc[:, 2]
    
    test_graph = generate_smile_graph(test_smi, Config.max_nodes)
    test_loader = get_torch_data(test_smi, test_y, test_graph, Config.batch_size, False)
    
    test_r2, test_rmse, test_mae = test(model, Config.device, test_loader, abeta_feature)
    print(f"Test results - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()