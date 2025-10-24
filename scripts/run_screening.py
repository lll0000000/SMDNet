#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual screening script entry point for SMDNet model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from config import Config
from src.screening import (
    mc_dropout_predictions, 
    compute_mutual_information, 
    candidate_selection, 
    reverse_sampling_with_mi
)
from src.model import CrossAttGATNet_AdaLN
from src.utils import generate_smile_graph, get_torch_data_unlabel, set_seed, compute_aac

def main():
    """Main virtual screening function"""
    print("Starting virtual screening...")
    
    # Set random seed for reproducibility
    set_seed(Config.seed)
    
    # Load training data (for distribution calculation)
    train_data = pd.read_csv(os.path.join(Config.data_dir, Config.train_file))
    trained_labels = train_data.iloc[:, 2]
    
    # Load unlabeled data for screening
    print("Loading screening library...")
    df_unlabel = pd.read_csv(os.path.join(Config.data_dir, 'BindingDB_data.csv'))
    unlabel_smi = df_unlabel.iloc[:, 0]
    unlabel_graph = generate_smile_graph(unlabel_smi, Config.max_nodes)
    unlabel_loader = get_torch_data_unlabel(unlabel_smi, unlabel_graph, Config.batch_size, False)
    
    # Compute Aβ protein features
    abeta_feature = compute_aac(Config.abeta_sequence)
    abeta_feature = torch.tensor(abeta_feature).float().to(Config.device)
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    model = CrossAttGATNet_AdaLN(
        max_nodes=Config.max_nodes,
        atom_feature_dim=Config.atom_feature_dim,
        embedding_dim=Config.embedding_dim,
        head=Config.head,
        fps_dim=Config.fps_dim, 
        abeta_dim=Config.abeta_dim,
        dropout=Config.dropout
    )
    model.load_state_dict(torch.load(os.path.join(Config.model_dir, Config.model_file), 
                                   map_location=Config.device))
    model = model.to(Config.device)
    
    # Compute pseudo-labels and uncertainties
    print("Computing pseudo-labels and uncertainties...")
    pseudo_label, var_label = mc_dropout_predictions(
        model, unlabel_loader, abeta_feature, Config.device, 
        num_samples=Config.mc_dropout_samples
    )
    pseudo_label = pseudo_label.flatten()
    var_label = var_label.flatten()
    
    # Compute mutual information
    mi = compute_mutual_information(var_label)
    
    # Save pseudo-label results
    df_pseudo = pd.DataFrame({
        'SMILES': unlabel_smi,
        'Pseudo_label': pseudo_label, 
        'Variance': var_label,
        'Mutual_Information': mi
    })
    df_pseudo.to_csv(os.path.join(Config.output_dir, 'pseudo_label_iter1.csv'), index=False)
    
    # Candidate set selection
    print("Selecting candidate set...")
    candidate_indices = candidate_selection(mi, candidate_percent=Config.candidate_percent)
    
    # Reverse sampling with mutual information
    pseudo_label_candidate = pseudo_label[candidate_indices]
    mi_candidate = mi[candidate_indices]
    
    selected_candidate_indices = reverse_sampling_with_mi(
        trained_labels, pseudo_label_candidate, mi_candidate, 
        num_samples=Config.num_samples, num_bins=Config.num_bins, 
        mi_weight=Config.mi_weight
    )
    final_selected_indices = candidate_indices[selected_candidate_indices]
    
    # Save screening results
    df_selected = pd.DataFrame({
        'Probe': np.nan,
        'SMILES': unlabel_smi.iloc[final_selected_indices].values,
        'pKd': pseudo_label[final_selected_indices],
        'Variance': var_label[final_selected_indices], 
        'Mutual_Information': mi[final_selected_indices]
    })
    df_selected.to_csv(os.path.join(Config.output_dir, 'selected_candidates_iter1.csv'), index=False)
    
    print(f"Screening completed! Selected {len(df_selected)} candidate molecules from {len(unlabel_smi)} total")
    print(f"Results saved to: {Config.output_dir}")

if __name__ == "__main__":
    main()