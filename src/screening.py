# -*- coding: utf-8 -*-
"""
Virtual screening functions with MC-Dropout uncertainty estimation and active learning.

Author: Yanling Wu

Date: August 2024
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        data_loader (DataLoader): Data loader for unlabeled molecules
        abeta_feature (torch.Tensor): Aβ protein features
        device (torch.device): Device for computation
        num_samples (int): Number of MC samples
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Mean predictions (num_data_points,)
            - Prediction variances (num_data_points,)
            
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        # Keep dropout active for uncertainty estimation
        model.train()
        
        predictions = []
        
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

        logger.info(f"MC-Dropout completed: {len(mean_pred)} predictions with uncertainty")
        return mean_pred, var_pred
        
    except Exception as e:
        raise RuntimeError(f"MC-Dropout prediction failed: {e}")


def compute_mutual_information(var_pred: np.ndarray, 
                              aleatoric_uncertainty: float = 0.1) -> np.ndarray:
    """
    Compute mutual information between predictions and model parameters.
    Lower MI indicates higher prediction confidence.
    
    Args:
        var_pred (np.ndarray): Predictive variance from MC-Dropout
        aleatoric_uncertainty (float): Aleatoric uncertainty parameter
        
    Returns:
        np.ndarray: Mutual information scores
    """
    try:
        # Ensure variance is non-negative
        var_pred = np.maximum(var_pred, 0)
        
        # Compute mutual information: I(Y; ω) = 0.5 * log(1 + σ² / (σ_noise² + ε))
        mutual_info = 0.5 * np.log(1 + var_pred / (aleatoric_uncertainty**2 + 1e-8))
        
        logger.debug(f"Computed mutual information: mean={np.mean(mutual_info):.4f}, "
                    f"min={np.min(mutual_info):.4f}, max={np.max(mutual_info):.4f}")
        return mutual_info
        
    except Exception as e:
        raise RuntimeError(f"Mutual information computation failed: {e}")


def candidate_selection(mi_scores: np.ndarray,
                       candidate_percent: Optional[float] = None,
                       mi_threshold: Optional[float] = None,
                       max_candidates: Optional[int] = None) -> np.ndarray:
    """
    Select candidate molecules based on mutual information scores.
    
    Args:
        mi_scores (np.ndarray): Mutual information scores
        candidate_percent (Optional[float]): Percentage of candidates to select
        mi_threshold (Optional[float]): MI threshold for selection
        max_candidates (Optional[int]): Maximum number of candidates
        
    Returns:
        np.ndarray: Indices of selected candidates
        
    Raises:
        ValueError: If selection criteria are invalid
    """
    try:
        if candidate_percent is not None:
            if not 0 < candidate_percent <= 1:
                raise ValueError(f"candidate_percent must be in (0, 1], got {candidate_percent}")
            
            num_candidates = int(len(mi_scores) * candidate_percent)
            if max_candidates is not None:
                num_candidates = min(num_candidates, max_candidates)
                
            candidate_indices = np.argsort(mi_scores)[:num_candidates]
            logger.info(f"Selected top {num_candidates} candidates ({candidate_percent*100:.1f}%) "
                       f"by mutual information")
            
        elif mi_threshold is not None:
            candidate_indices = np.where(mi_scores < mi_threshold)[0]
            logger.info(f"Selected {len(candidate_indices)} candidates with MI < {mi_threshold}")
            
        else:
            raise ValueError("Must provide either candidate_percent or mi_threshold")
        
        if len(candidate_indices) == 0:
            logger.warning("No candidates selected. Consider relaxing selection criteria.")
        
        return candidate_indices
        
    except Exception as e:
        raise RuntimeError(f"Candidate selection failed: {e}")


def compute_bin_distribution(labels: np.ndarray, 
                           bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distribution of labels across bins and assign bin indices.
    
    Args:
        labels (np.ndarray): Array of label values
        bins (np.ndarray): Bin edges
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Bin counts (len(bins)-1,)
            - Bin indices for each sample
    """
    try:
        # Digitize labels to bins (returns bin indices starting from 1)
        bin_indices = np.digitize(labels, bins) - 1
        
        # Ensure indices are within valid range
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        # Compute bin counts
        bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
        
        return bin_counts, bin_indices
        
    except Exception as e:
        raise RuntimeError(f"Bin distribution computation failed: {e}")


def reverse_sampling_with_mi(trained_labels: np.ndarray,
                            pseudo_labels: np.ndarray,
                            mi_scores: np.ndarray,
                            num_samples: int,
                            num_bins: int,
                            mi_weight: float = 0.5) -> np.ndarray:
    """
    Perform reverse sampling with mutual information weighting.
    
    Args:
        trained_labels (np.ndarray): Labels from training data
        pseudo_labels (np.ndarray): Pseudo-labels from model predictions
        mi_scores (np.ndarray): Mutual information scores
        num_samples (int): Number of samples to select
        num_bins (int): Number of bins for distribution analysis
        mi_weight (float): Weight for MI vs distribution sampling (0-1)
        
    Returns:
        np.ndarray: Indices of selected samples
        
    Raises:
        ValueError: If input validation fails
    """
    try:
        if len(pseudo_labels) != len(mi_scores):
            raise ValueError("Pseudo labels and MI scores must have same length")
        
        if num_samples > len(pseudo_labels):
            logger.warning(f"Requested {num_samples} samples but only {len(pseudo_labels)} available")
            num_samples = len(pseudo_labels)
        
        # Create bins based on training data distribution
        bins = np.linspace(np.min(trained_labels), np.max(trained_labels), num_bins + 1)
        
        # Compute training set distribution
        trained_bin_count, _ = compute_bin_distribution(trained_labels, bins)
        
        # Compute pseudo-label distribution
        pseudo_bin_count, pseudo_bin_indices = compute_bin_distribution(pseudo_labels, bins)
        
        # Reorder training frequencies to prioritize rare regions
        sorted_indices = np.argsort(trained_bin_count)  # Ascending order (rare to frequent)
        trained_bin_count_prime = np.zeros_like(trained_bin_count)
        
        for rank, idx in enumerate(sorted_indices):
            # Reverse the order: assign frequent bin counts to rare bin positions
            trained_bin_count_prime[idx] = trained_bin_count[sorted_indices[-(rank + 1)]]
        
        # Compute reverse sampling probabilities
        max_count = np.max(trained_bin_count)
        reverse_sample_rate = trained_bin_count_prime / (max_count + 1e-8)
        
        # Apply reverse sampling to pseudo-label data
        sampling_probs = np.array([reverse_sample_rate[i] for i in pseudo_bin_indices])
        
        # Normalize MI scores for weighting
        mi_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
        mi_probs = 1 - mi_norm  # Lower MI → higher confidence → higher weight
        
        # Combine sampling probabilities with MI weights
        final_sampling_probs = (mi_weight * mi_probs + 
                               (1 - mi_weight) * sampling_probs)
        
        # Ensure probabilities are valid
        final_sampling_probs = np.maximum(final_sampling_probs, 0)
        final_sampling_probs /= (np.sum(final_sampling_probs) + 1e-8)
        
        # Select top samples by probability
        top_indices = np.argsort(final_sampling_probs)[-num_samples:]
        
        logger.info(f"Reverse sampling completed: selected {len(top_indices)} samples "
                   f"(MI weight: {mi_weight})")
        
        return top_indices
        
    except Exception as e:
        raise RuntimeError(f"Reverse sampling failed: {e}")


def virtual_screening_pipeline(model: nn.Module,
                              unlabeled_smiles: List[str],
                              trained_labels: np.ndarray,
                              abeta_feature: torch.Tensor,
                              device: torch.device,
                              max_nodes: int = 75,
                              batch_size: int = 32,
                              candidate_percent: float = 0.4,
                              num_samples: int = 100,
                              num_bins: int = 5,
                              mi_weight: float = 0.5,
                              mc_samples: int = 10,
                              output_dir: str = 'output') -> Dict[str, Any]:
    """
    Complete virtual screening pipeline with uncertainty-aware candidate selection.
    
    Args:
        model (nn.Module): Trained model
        unlabeled_smiles (List[str]): SMILES strings for screening
        trained_labels (np.ndarray): Training data labels for distribution matching
        abeta_feature (torch.Tensor): Aβ protein features
        device (torch.device): Device for computation
        max_nodes (int): Maximum atoms per molecule
        batch_size (int): Batch size for inference
        candidate_percent (float): Percentage for initial candidate selection
        num_samples (int): Final number of samples to select
        num_bins (int): Number of bins for distribution analysis
        mi_weight (float): Weight for mutual information in sampling
        mc_samples (int): Number of MC-Dropout samples
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, Any]: Screening results and metadata
    """
    try:
        from src.utils import generate_smile_graph, get_torch_data_unlabel
        
        logger.info(f"Starting virtual screening for {len(unlabeled_smiles)} molecules")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate molecular graphs
        logger.info("Generating molecular graphs...")
        unlabel_graph = generate_smile_graph(unlabeled_smiles, max_nodes)
        unlabel_loader = get_torch_data_unlabel(unlabeled_smiles, unlabel_graph, batch_size, False)
        
        # Compute pseudo-labels and uncertainties
        logger.info("Computing pseudo-labels with MC-Dropout...")
        pseudo_label, var_label = mc_dropout_predictions(
            model, unlabel_loader, abeta_feature, device, num_samples=mc_samples
        )
        pseudo_label = pseudo_label.flatten()
        var_label = var_label.flatten()
        
        # Compute mutual information
        mi_scores = compute_mutual_information(var_label)
        
        # Save initial screening results
        screening_results = pd.DataFrame({
            'SMILES': unlabeled_smiles,
            'Pseudo_label': pseudo_label,
            'Variance': var_label,
            'Mutual_Information': mi_scores
        })
        
        screening_file = os.path.join(output_dir, 'initial_screening_results.csv')
        screening_results.to_csv(screening_file, index=False)
        logger.info(f"Initial screening results saved to {screening_file}")
        
        # Initial candidate selection
        candidate_indices = candidate_selection(mi_scores, candidate_percent=candidate_percent)
        
        if len(candidate_indices) == 0:
            logger.error("No candidates selected in initial screening")
            return {
                'success': False,
                'error': 'No candidates selected',
                'screening_results': screening_results
            }
        
        # Refined sampling within candidates
        pseudo_label_candidate = pseudo_label[candidate_indices]
        mi_candidate = mi_scores[candidate_indices]
        
        selected_candidate_indices = reverse_sampling_with_mi(
            trained_labels, pseudo_label_candidate, mi_candidate,
            num_samples=num_samples, num_bins=num_bins, mi_weight=mi_weight
        )
        
        # Map back to original indices
        final_selected_indices = candidate_indices[selected_candidate_indices]
        
        # Create final selection dataframe
        df_selected = pd.DataFrame({
            'Probe': np.nan,
            'SMILES': np.array(unlabeled_smiles)[final_selected_indices],
            'pKd': pseudo_label[final_selected_indices],
            'Variance': var_label[final_selected_indices],
            'Mutual_Information': mi_scores[final_selected_indices]
        })
        
        selected_file = os.path.join(output_dir, 'selected_candidates.csv')
        df_selected.to_csv(selected_file, index=False)
        
        # Generate screening report
        report = {
            'total_molecules': len(unlabeled_smiles),
            'initial_candidates': len(candidate_indices),
            'final_selection': len(df_selected),
            'pseudo_label_stats': {
                'mean': np.mean(pseudo_label),
                'std': np.std(pseudo_label),
                'min': np.min(pseudo_label),
                'max': np.max(pseudo_label)
            },
            'mi_stats': {
                'mean': np.mean(mi_scores),
                'std': np.std(mi_scores),
                'min': np.min(mi_scores),
                'max': np.max(mi_scores)
            },
            'output_files': {
                'screening_results': screening_file,
                'selected_candidates': selected_file
            }
        }
        
        logger.info(f"Virtual screening completed: "
                   f"{report['final_selection']} molecules selected from {report['total_molecules']}")
        
        return {
            'success': True,
            'report': report,
            'screening_results': screening_results,
            'selected_candidates': df_selected,
            'final_indices': final_selected_indices
        }
        
    except Exception as e:
        logger.error(f"Virtual screening pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def create_mixed_dataset(original_train_data: pd.DataFrame,
                        selected_candidates: pd.DataFrame,
                        output_dir: str = 'output') -> pd.DataFrame:
    """
    Create mixed dataset by combining original training data with selected candidates.
    
    Args:
        original_train_data (pd.DataFrame): Original training data
        selected_candidates (pd.DataFrame): Selected candidate molecules
        output_dir (str): Output directory
        
    Returns:
        pd.DataFrame: Mixed dataset
    """
    try:
        # Prepare selected candidates (remove uncertainty columns, add source)
        df_selected_clean = selected_candidates[['SMILES', 'pKd']].copy()
        df_selected_clean['Source'] = 'Pseudo'
        
        # Ensure column consistency
        if 'Source' not in original_train_data.columns:
            original_train_data = original_train_data.copy()
            original_train_data['Source'] = 'Real'
        
        # Combine datasets
        mixed_data = pd.concat([original_train_data, df_selected_clean], ignore_index=True)
        
        # Save mixed dataset
        mixed_file = os.path.join(output_dir, 'mixed_training_data.csv')
        mixed_data.to_csv(mixed_file, index=False)
        
        logger.info(f"Mixed dataset created: {len(original_train_data)} real + "
                   f"{len(selected_candidates)} pseudo = {len(mixed_data)} total")
        
        return mixed_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to create mixed dataset: {e}")


def update_screening_pool(original_pool: pd.DataFrame,
                         selected_smiles: List[str],
                         output_dir: str = 'output') -> pd.DataFrame:
    """
    Update screening pool by removing selected molecules.
    
    Args:
        original_pool (pd.DataFrame): Original screening pool
        selected_smiles (List[str]): SMILES of selected molecules
        output_dir (str): Output directory
        
    Returns:
        pd.DataFrame: Updated screening pool
    """
    try:
        # Remove selected molecules
        updated_pool = original_pool[~original_pool['SMILES'].isin(selected_smiles)].copy()
        
        # Save updated pool
        pool_file = os.path.join(output_dir, 'updated_screening_pool.csv')
        updated_pool.to_csv(pool_file, index=False)
        
        logger.info(f"Screening pool updated: {len(original_pool)} → {len(updated_pool)} "
                   f"({len(selected_smiles)} molecules removed)")
        
        return updated_pool
        
    except Exception as e:
        raise RuntimeError(f"Failed to update screening pool: {e}")


# Convenience function for configuration-based screening
def screening_from_config(config,
                         model: nn.Module,
                         unlabeled_smiles: List[str],
                         trained_labels: np.ndarray,
                         abeta_feature: torch.Tensor,
                         iteration: int = 1) -> Dict[str, Any]:
    """
    Run virtual screening using configuration parameters.
    
    Args:
        config: Configuration object
        model (nn.Module): Trained model
        unlabeled_smiles (List[str]): SMILES for screening
        trained_labels (np.ndarray): Training labels
        abeta_feature (torch.Tensor): Aβ protein features
        iteration (int): Screening iteration number
        
    Returns:
        Dict[str, Any]: Screening results
    """
    logger.info(f"Starting virtual screening iteration {iteration}")
    
    # Create iteration-specific output directory
    output_dir = os.path.join(config.output_dir, f'iteration_{iteration}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run screening pipeline
    results = virtual_screening_pipeline(
        model=model,
        unlabeled_smiles=unlabeled_smiles,
        trained_labels=trained_labels,
        abeta_feature=abeta_feature,
        device=config.device,
        max_nodes=config.max_nodes,
        batch_size=config.batch_size,
        candidate_percent=config.candidate_percent,
        num_samples=config.num_samples,
        num_bins=config.num_bins,
        mi_weight=config.mi_weight,
        mc_samples=config.mc_dropout_samples,
        output_dir=output_dir
    )
    
    if results['success']:
        logger.info(f"Screening iteration {iteration} completed successfully")
        
        # Save detailed report
        report_file = os.path.join(output_dir, 'screening_report.json')
        import json
        with open(report_file, 'w') as f:
            json.dump(results['report'], f, indent=2)
            
        logger.info(f"Screening report saved to {report_file}")
    else:
        logger.error(f"Screening iteration {iteration} failed: {results.get('error', 'Unknown error')}")
    
    return results