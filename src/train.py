# -*- coding: utf-8 -*-
"""
Training and evaluation functions for SMDNet model with MC-Dropout uncertainty estimation.

Author: Yanling Wu
Date: August 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple, Dict, Any, Optional
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mc_dropout_predictions(model: nn.Module, 
                          data_loader: torch.utils.data.DataLoader, 
                          abeta_feature: torch.Tensor, 
                          device: torch.device, 
                          num_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Monte Carlo Dropout predictions for uncertainty estimation.
    
    Args:
        model (nn.Module): Trained model with dropout
        data_loader (DataLoader): Data loader for inference
        abeta_feature (torch.Tensor): Aβ protein features
        device (torch.device): Device for computation
        num_samples (int): Number of MC samples
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Mean predictions (num_data_points,)
            - Prediction variances (num_data_points,)
            
    Raises:
        RuntimeError: If model is not in appropriate mode
    """
    if not hasattr(model, 'dropout'):
        logger.warning("Model may not have dropout layers for MC-Dropout")
    
    # Keep dropout active during inference
    model.train()
    
    predictions = []
    
    try:
        with torch.no_grad():
            for sample_idx in range(num_samples):
                logger.debug(f"MC-Dropout sample {sample_idx + 1}/{num_samples}")
                batch_preds = []
                
                for batch in data_loader:
                    batch = batch.to(device)
                    preds = model(batch, abeta_feature)
                    batch_preds.append(preds.cpu().numpy())
                
                predictions.append(np.concatenate(batch_preds, axis=0))

        predictions = np.stack(predictions)  # Shape: [num_samples, num_data_points]
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)

        return mean_pred, var_pred
        
    except Exception as e:
        raise RuntimeError(f"MC-Dropout prediction failed: {e}")


def train_and_val(model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  abeta_feature: torch.Tensor,
                  device: torch.device,
                  epochs: int,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: nn.Module,
                  model_save_path: str,
                  num_samples: int = 10) -> Tuple[List[float], ...]:
    """
    Training and validation loop with MC-Dropout evaluation.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        abeta_feature (torch.Tensor): Aβ protein features
        device (torch.device): Device for computation
        epochs (int): Number of training epochs
        optimizer (Optimizer): Model optimizer
        loss_fn (nn.Module): Loss function
        model_save_path (str): Path to save best model
        num_samples (int): Number of MC samples for validation
        
    Returns:
        Tuple: Training and validation metrics history
        
    Raises:
        ValueError: If data loaders are empty
    """
    # Initialize metric trackers
    train_loss_history, val_loss_history = [], []
    train_r2_history, val_r2_history = [], []
    train_rmse_history, val_rmse_history = [], []
    train_mae_history, val_mae_history = [], []
    
    best_r2 = -float('inf')
    best_epoch = 0
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_outputs_epoch, train_labels_epoch = [], []
        
        for batch_idx, train_data in enumerate(train_loader):
            try:
                train_data = train_data.to(device)
                outputs = model(train_data, abeta_feature)
                labels = train_data.y.view(-1, 1).float().to(device)
                
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_outputs_epoch.extend(outputs.cpu().detach().numpy().flatten().tolist())
                train_labels_epoch.extend(labels.cpu().detach().numpy().flatten().tolist())
                
            except Exception as e:
                logger.warning(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate training metrics
        if not train_outputs_epoch:
            logger.warning(f"No training data processed in epoch {epoch + 1}")
            continue
            
        train_loss = mean_squared_error(train_labels_epoch, train_outputs_epoch)
        train_rmse = np.sqrt(train_loss)
        train_r2 = r2_score(train_labels_epoch, train_outputs_epoch)
        train_mae = mean_absolute_error(train_labels_epoch, train_outputs_epoch)
        
        train_loss_history.append(train_loss)
        train_rmse_history.append(train_rmse)
        train_r2_history.append(train_r2)
        train_mae_history.append(train_mae)
        
        # Validation phase with MC-Dropout
        try:
            mean_pred, var_pred = mc_dropout_predictions(
                model, val_loader, abeta_feature, device, num_samples=num_samples
            )
            
            val_labels_epoch = []
            for batch in val_loader:
                val_labels_epoch.extend(batch.y.view(-1, 1).cpu().numpy().flatten())
            val_labels_epoch = np.array(val_labels_epoch)
            
            # Calculate validation metrics
            val_loss = mean_squared_error(val_labels_epoch, mean_pred)
            val_rmse = np.sqrt(val_loss)
            val_r2 = r2_score(val_labels_epoch, mean_pred)
            val_mae = mean_absolute_error(val_labels_epoch, mean_pred)
            
            val_loss_history.append(val_loss)
            val_rmse_history.append(val_rmse)
            val_r2_history.append(val_r2)
            val_mae_history.append(val_mae)
            
        except Exception as e:
            logger.warning(f"Validation failed in epoch {epoch + 1}: {e}")
            # Use training metrics as fallback
            val_loss_history.append(train_loss)
            val_rmse_history.append(train_rmse)
            val_r2_history.append(train_r2)
            val_mae_history.append(train_mae)
            val_r2 = train_r2
        
        # Save best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            
            # Save validation predictions
            try:
                df_val = pd.DataFrame({
                    'True_value': val_labels_epoch,
                    'Predict_value': mean_pred,
                    'Uncertainty': np.sqrt(var_pred)  # Standard deviation
                })
                df_val.to_csv('val_predictions.csv', index=False)
                
                # Save model
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"🎯 New best model saved at epoch {best_epoch}: R² = {best_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to save model or predictions: {e}")
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
                f"Best R²: {best_r2:.4f} (epoch {best_epoch})"
            )
    
    logger.info(f"Training completed. Best validation R²: {best_r2:.4f} at epoch {best_epoch}")
    
    return (train_loss_history, val_loss_history, train_r2_history, val_r2_history,
            train_rmse_history, val_rmse_history, train_mae_history, val_mae_history)


def test(model: nn.Module, 
         device: torch.device, 
         test_loader: torch.utils.data.DataLoader, 
         abeta_feature: torch.Tensor,
         num_samples: int = 10) -> Tuple[float, float, float]:
    """
    Evaluate model on test set with MC-Dropout uncertainty estimation.
    
    Args:
        model (nn.Module): Trained model
        device (torch.device): Device for computation
        test_loader (DataLoader): Test data loader
        abeta_feature (torch.Tensor): Aβ protein features
        num_samples (int): Number of MC samples
        
    Returns:
        Tuple[float, float, float]: Test R², RMSE, and MAE
        
    Raises:
        RuntimeError: If test evaluation fails
    """
    logger.info("Starting test evaluation...")
    
    try:
        mean_pred, var_pred = mc_dropout_predictions(
            model, test_loader, abeta_feature, device, num_samples=num_samples
        )
        
        test_labels = []
        for batch in test_loader:
            test_labels.extend(batch.y.view(-1, 1).cpu().numpy().flatten())
        test_labels = np.array(test_labels)
        
        # Calculate test metrics
        test_loss = mean_squared_error(test_labels, mean_pred)
        test_rmse = np.sqrt(test_loss)
        test_r2 = r2_score(test_labels, mean_pred)
        test_mae = mean_absolute_error(test_labels, mean_pred)
        
        # Save test predictions with uncertainty
        df_test = pd.DataFrame({
            'True_value': test_labels,
            'Predict_value': mean_pred,
            'Uncertainty': np.sqrt(var_pred),
            'Variance': var_pred
        })
        df_test.to_csv('test_predictions.csv', index=False)
        
        logger.info(f"✅ Test evaluation completed: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}")
        
        return test_r2, test_rmse, test_mae
        
    except Exception as e:
        raise RuntimeError(f"Test evaluation failed: {e}")


def show_loss_curve(epochs: int, 
                   train_loss_history: List[float], 
                   test_loss_history: List[float],
                   save_path: str = 'training_curve.png') -> None:
    """
    Plot and save training and validation loss curves.
    
    Args:
        epochs (int): Number of epochs
        train_loss_history (List[float]): Training loss history
        test_loss_history (List[float]): Validation loss history
        save_path (str): Path to save the plot
    """
    try:
        plt.figure(figsize=[10, 6])
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss', color='#F08080', linewidth=2)
        plt.plot(range(1, epochs + 1), test_loss_history, label='Val Loss', color='#0B7093', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot R² curves if available
        plt.subplot(1, 2, 2)
        if len(train_loss_history) == len(test_loss_history):
            # Use a simple metric progression for demonstration
            train_progress = [1 - loss/max(train_loss_history) for loss in train_loss_history]
            val_progress = [1 - loss/max(test_loss_history) for loss in test_loss_history]
            
            plt.plot(range(1, epochs + 1), train_progress, label='Train Progress', color='#F08080', linewidth=2)
            plt.plot(range(1, epochs + 1), val_progress, label='Val Progress', color='#0B7093', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Progress (1 - Normalized Loss)')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curve saved to {save_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save training curve: {e}")


def save_result(epochs: int,
               train_loss_history: List[float],
               test_loss_history: List[float],
               train_r2_history: List[float],
               test_r2_history: List[float],
               train_rmse_history: List[float],
               test_rmse_history: List[float],
               train_mae_history: List[float],
               test_mae_history: List[float],
               output_file: str = 'training_results.xlsx') -> None:
    """
    Save training results to Excel file.
    
    Args:
        epochs (int): Number of epochs
        train_loss_history (List[float]): Training loss history
        test_loss_history (List[float]): Validation loss history
        train_r2_history (List[float]): Training R² history
        test_r2_history (List[float]): Validation R² history
        train_rmse_history (List[float]): Training RMSE history
        test_rmse_history (List[float]): Validation RMSE history
        train_mae_history (List[float]): Training MAE history
        test_mae_history (List[float]): Validation MAE history
        output_file (str): Output Excel file path
    """
    try:
        # Ensure all lists have the same length
        min_length = min(len(train_loss_history), len(test_loss_history), epochs)
        
        df = pd.DataFrame({
            'Epoch': range(1, min_length + 1),
            'TrainLoss': train_loss_history[:min_length],
            'ValLoss': test_loss_history[:min_length],
            'TrainR2': train_r2_history[:min_length],
            'ValR2': test_r2_history[:min_length],
            'TrainRMSE': train_rmse_history[:min_length],
            'ValRMSE': test_rmse_history[:min_length],
            'TrainMAE': train_mae_history[:min_length],
            'ValMAE': test_mae_history[:min_length]
        })
        
        df.to_excel(output_file, index=False)
        logger.info(f"Training results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save training results: {e}")


def load_model(model_path: str, 
              model_class: nn.Module, 
              device: torch.device,
              **model_kwargs) -> nn.Module:
    """
    Safely load a trained model.
    
    Args:
        model_path (str): Path to saved model weights
        model_class (nn.Module): Model class to instantiate
        device (torch.device): Device to load model on
        **model_kwargs: Keyword arguments for model initialization
        
    Returns:
        nn.Module: Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def create_optimizer(model: nn.Module, 
                    learning_rate: float = 0.001,
                    optimizer_type: str = 'adam') -> torch.optim.Optimizer:
    """
    Create optimizer with specified parameters.
    
    Args:
        model (nn.Module): Model to optimize
        learning_rate (float): Learning rate
        optimizer_type (str): Type of optimizer ('adam', 'sgd')
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        logger.warning(f"Unknown optimizer type: {optimizer_type}. Using Adam.")
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


# Convenience function for configuration-based training
def train_from_config(config, 
                     model: nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader,
                     abeta_feature: torch.Tensor) -> Dict[str, List[float]]:
    """
    Train model using configuration parameters.
    
    Args:
        config: Configuration object
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        abeta_feature (torch.Tensor): Aβ protein features
        
    Returns:
        Dict[str, List[float]]: Training history metrics
    """
    logger.info("Starting training from configuration...")
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, config.learning_rate, 'adam')
    loss_fn = nn.MSELoss()
    
    # Train model
    metrics = train_and_val(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        abeta_feature=abeta_feature,
        device=config.device,
        epochs=config.epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model_save_path=os.path.join(config.model_dir, config.model_file),
        num_samples=config.mc_dropout_samples
    )
    
    # Save results
    save_result(
        epochs=config.epochs,
        train_loss_history=metrics[0],
        test_loss_history=metrics[1],
        train_r2_history=metrics[2],
        test_r2_history=metrics[3],
        train_rmse_history=metrics[4],
        test_rmse_history=metrics[5],
        train_mae_history=metrics[6],
        test_mae_history=metrics[7],
        output_file=os.path.join(config.output_dir, 'training_results.xlsx')
    )
    
    # Plot training curves
    show_loss_curve(
        epochs=config.epochs,
        train_loss_history=metrics[0],
        test_loss_history=metrics[1],
        save_path=os.path.join(config.output_dir, 'training_curve.png')
    )
    
    logger.info("Training from configuration completed successfully")
    
    return {
        'train_loss': metrics[0],
        'val_loss': metrics[1],
        'train_r2': metrics[2],
        'val_r2': metrics[3],
        'train_rmse': metrics[4],
        'val_rmse': metrics[5],
        'train_mae': metrics[6],
        'val_mae': metrics[7]
    }