# -*- coding: utf-8 -*-
"""
Optimized Graph Attention Network with Cross-Attention and Adaptive Layer Normalization
for molecular property prediction.

Author: Yanling Wu
Date: August 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
import warnings
warnings.filterwarnings('ignore')


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for feature interaction between molecular graph and fingerprint features.
    """
    
    def __init__(self, embed_dim: int):
        """
        Initialize cross-attention layer.
        
        Args:
            embed_dim (int): Embedding dimension for query, key, and value projections
        """
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def _apply_mask(self, att_scores: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to attention scores for padded nodes.
        
        Args:
            att_scores (torch.Tensor): Attention scores (B, Q, K)
            key_mask (torch.Tensor): Boolean mask for key positions (B, K), True=keep, False=mask
        
        Returns:
            torch.Tensor: Masked attention scores
        """
        if key_mask is None:
            return att_scores
        
        # Set masked positions to -inf so they become 0 after softmax
        minus_inf = torch.finfo(att_scores.dtype).min
        att_scores = att_scores.masked_fill(~key_mask[:, None, :], minus_inf)
        return att_scores

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                key_mask: torch.Tensor = None, 
                return_attn: bool = False):
        """
        Forward pass for cross-attention.
        
        Args:
            q (torch.Tensor): Query tensor (B, Q, D)
            k (torch.Tensor): Key tensor (B, K, D)
            v (torch.Tensor): Value tensor (B, K, D)
            key_mask (torch.Tensor, optional): Boolean mask for key positions (B, K)
            return_attn (bool): Whether to return attention weights
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Attended features or (attended features, attention weights)
        """
        Q = self.query_proj(q)
        K = self.key_proj(k)
        V = self.value_proj(v)

        # Compute attention scores
        att_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # (B, Q, K)
        att_scores = self._apply_mask(att_scores, key_mask)
        att_weights = self.softmax(att_scores)  # (B, Q, K)

        # Apply attention to values
        attended = torch.matmul(att_weights, V)  # (B, Q, D)
        
        return (attended, att_weights) if return_attn else attended


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) that conditions normalization 
    on protein features (Aβ sequence).
    """
    
    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-6):
        """
        Initialize Adaptive Layer Normalization.
        
        Args:
            normalized_shape (int): Feature dimension for normalization
            cond_dim (int): Condition input dimension (protein features)
            eps (float): Epsilon for numerical stability
        """
        super(AdaptiveLayerNorm, self).__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # Learnable γ
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # Learnable β
        
        # Condition modulation networks (Aβ protein features)
        self.fc_gamma = nn.Linear(cond_dim, normalized_shape)  # Compute Δγ
        self.fc_beta = nn.Linear(cond_dim, normalized_shape)   # Compute Δβ
        
    def forward(self, x: torch.Tensor, cond_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional normalization.
        
        Args:
            x (torch.Tensor): Input features (batch_size, feature_dim)
            cond_input (torch.Tensor): Condition features (batch_size, cond_dim)
        
        Returns:
            torch.Tensor: Normalized and conditioned features
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Compute condition-dependent shifts
        delta_gamma = self.fc_gamma(cond_input)
        delta_beta = self.fc_beta(cond_input)
        
        # Apply normalization with condition modulation
        x_norm = (x - mean) / (std + self.eps)
        gamma = self.gamma + delta_gamma
        beta = self.beta + delta_beta
        
        return gamma * x_norm + beta


class CrossAttGATNet_AdaLN(nn.Module):
    """
    Main model: Graph Attention Network with Cross-Attention and Adaptive Layer Normalization
    for predicting molecular binding affinity with Aβ protein.
    """
    
    def __init__(self, 
                 max_nodes: int,
                 atom_feature_dim: int, 
                 embedding_dim: int,
                 head: int,
                 fps_dim: int,
                 abeta_dim: int,
                 dropout: float,
                 n_output: int = 1,
                 output_dim: int = 128):
        """
        Initialize the complete model architecture.
        
        Args:
            max_nodes (int): Maximum number of atoms per molecule
            atom_feature_dim (int): Dimension of atom features
            embedding_dim (int): GAT embedding dimension
            head (int): Number of attention heads
            fps_dim (int): Molecular fingerprint dimension
            abeta_dim (int): Aβ protein feature dimension
            dropout (float): Dropout rate
            n_output (int): Number of output predictions
            output_dim (int): Output dimension after GAT layers
        """
        super(CrossAttGATNet_AdaLN, self).__init__()
       
        # Store model parameters
        self.atom_feature_dim = atom_feature_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        self.head = head
        self.fps_dim = fps_dim
        self.abeta_dim = abeta_dim
        self.dropout_rate = dropout          
       
        # GAT Layers for molecular graph processing
        self.gat1 = GATConv(atom_feature_dim, embedding_dim, dropout=dropout)
        self.gat2 = GATConv(embedding_dim, embedding_dim, heads=head, dropout=dropout)
        self.gat3 = GATConv(embedding_dim * head, output_dim, dropout=dropout)
       
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim * head)
        self.bn3 = nn.BatchNorm1d(output_dim)
       
        # MLP layers for fingerprint processing
        self.fc_fp1 = nn.Linear(fps_dim, embedding_dim)
        self.fc_fp2 = nn.Linear(embedding_dim, output_dim)
       
        # Cross-attention layers for feature interaction
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(output_dim) for _ in range(2)
        ])

        # Adaptive Layer Normalization with protein conditioning
        self.ada_ln_x = AdaptiveLayerNorm(output_dim, abeta_dim)
        self.ada_ln_fps = AdaptiveLayerNorm(output_dim, abeta_dim)
       
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)  

        # Activation and regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, 
                data: torch.Tensor, 
                abeta_feature: torch.Tensor, 
                return_attn: bool = False):
        """
        Forward pass through the complete model.
        
        Args:
            data: PyTorch Geometric data object containing:
                - x: Atom features
                - edge_index: Graph connectivity
                - batch: Batch indices
                - fps: Molecular fingerprints
            abeta_feature (torch.Tensor): Aβ protein features (batch_size, abeta_dim)
            return_attn (bool): Whether to return attention weights for interpretation
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict]]: 
                Predictions or (predictions, attention dictionary)
        """
        # Extract data components
        x, edge_index, batch, fps = data.x, data.edge_index, data.batch, data.fps
               
        x_in = x 
        batch_size = batch.max().item() + 1
       
        # Reshape for cross-attention
        x_in_dense = x_in.view(batch_size, self.max_nodes, self.atom_feature_dim)
        node_mask = (x_in_dense.abs().sum(dim=-1) > 0)  # (B, max_nodes) bool mask

        # Process molecular graph with GAT layers
        x = self.dropout(self.relu(self.bn1(self.gat1(x, edge_index))))
        x = self.dropout(self.relu(self.bn2(self.gat2(x, edge_index))))
        x = self.dropout(self.relu(self.bn3(self.gat3(x, edge_index))))  # (N_total, D)

        # Process molecular fingerprints
        embedded_fps = self.dropout(self.relu(self.fc_fp1(fps)))
        embedded_fps = self.dropout(self.relu(self.fc_fp2(embedded_fps)))  # (B, D)

        # ===== Cross-Attention Mechanism =====
        viewed_x = x.view(batch_size, self.max_nodes, self.output_dim)    # (B, N, D)
        viewed_fps = embedded_fps.view(batch_size, 1, self.output_dim)    # (B, 1, D)

        # Cross-attention: nodes ← fingerprints
        attended_x = self.cross_attention_layers[0](
            viewed_x, viewed_fps, viewed_fps, key_mask=None, return_attn=False
        )  # (B, N, D)
        attended_x = attended_x.view(-1, self.output_dim)

        # Cross-attention: fingerprints ← nodes
        out = self.cross_attention_layers[1](
            viewed_fps, viewed_x, viewed_x, key_mask=node_mask, return_attn=return_attn
        )
       
        if return_attn:
            attended_fps, att_fps_to_nodes = out  # (B, 1, D), (B, 1, N)
        else:
            attended_fps = out
            att_fps_to_nodes = None

        attended_fps = attended_fps.view(-1, self.output_dim)            
        
        # ===== Adaptive Layer Normalization with Skip Connections =====
        x = self.ada_ln_x(x + attended_x.view(-1, self.output_dim), abeta_feature)
        embedded_fps = self.ada_ln_fps(embedded_fps + attended_fps, abeta_feature)

        # Global pooling and final prediction
        x = gmp(x, batch)  # Global max pooling over molecules
        xc = torch.cat((x, embedded_fps), 1)  # Concatenate graph and fingerprint features
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        out = self.out(xc)
        
        # Return attention weights if requested (for interpretability)
        if return_attn:
            # Process attention weights for interpretation
            att = att_fps_to_nodes.squeeze(1)  # (B, N)
            valid = node_mask.bool()  # (B, N)
        
            # Normalize attention weights
            att_max = att.masked_fill(~valid, -1e9).max(dim=1, keepdim=True).values
            att_min = att.masked_fill(~valid, 1e9).min(dim=1, keepdim=True).values
            denom = (att_max - att_min).clamp_min(1e-8)
        
            att = (att - att_min) / denom       
            att = torch.where(valid, att, torch.zeros_like(att))
            att = torch.clamp(att, 0.0, 1.0)
        
            return out, {"fps_to_nodes": att, "node_mask": node_mask}
        else:
            return out


# Convenience function for model creation
def create_model(config) -> CrossAttGATNet_AdaLN:
    """
    Create model instance from configuration.
    
    Args:
        config: Configuration object with model parameters
    
    Returns:
        CrossAttGATNet_AdaLN: Initialized model
    """
    return CrossAttGATNet_AdaLN(
        max_nodes=config.max_nodes,
        atom_feature_dim=config.atom_feature_dim,
        embedding_dim=config.embedding_dim,
        head=config.head,
        fps_dim=config.fps_dim,
        abeta_dim=config.abeta_dim,
        dropout=config.dropout,
        n_output=config.n_output,
        output_dim=config.output_dim
    )