# -*- coding: utf-8 -*-
"""
Model interpretability and visualization functions for atom-level importance analysis.

Author: Your Name
Date: August 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import logging
from rdkit import Chem
from rdkit.Chem import Draw
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def color_atoms_on_mol(mol: Chem.Mol, 
                      scores: np.ndarray, 
                      gamma: float = 0.7) -> Optional[bytes]:
    """
    Color atoms based on importance scores with blue-to-red colormap.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        scores (np.ndarray): Atom importance scores (num_atoms,)
        gamma (float): Contrast enhancement parameter (<1 enhances high values)
        
    Returns:
        Optional[bytes]: PNG image bytes or None if failed
        
    Raises:
        ValueError: If scores dimension doesn't match number of atoms
    """
    try:
        if len(scores) != mol.GetNumAtoms():
            raise ValueError(f"Score length ({len(scores)}) doesn't match number of atoms ({mol.GetNumAtoms()})")
        
        scores = np.asarray(scores, dtype=float)
        
        # Normalize scores to [0, 1]
        s_min = float(scores.min())
        s_max = float(scores.max())
        rng = s_max - s_min
        
        if rng < 1e-8:
            # All scores are similar, apply small perturbation to avoid uniform coloring
            scores = scores - s_min
            logger.debug("All scores are similar, applying uniform coloring with perturbation")
        else:
            scores = (scores - s_min) / (rng + 1e-8)

        # Apply gamma correction for contrast enhancement
        scores = np.power(scores, gamma)

        # Create color mapping (blue to red)
        colors = {}
        for idx in range(mol.GetNumAtoms()):
            val = float(scores[idx])  # [0, 1]
            # Blue (low importance) to Red (high importance)
            colors[idx] = (1.0, 1 - val, 1 - val)
        
        # Generate molecular image
        d2d = Draw.MolDraw2DCairo(600, 400)
        d2d.DrawMolecule(
            mol, 
            highlightAtoms=list(range(mol.GetNumAtoms())), 
            highlightAtomColors=colors,
            highlightBonds=[], 
            highlightBondColors={}
        )
        d2d.FinishDrawing()
        png = d2d.GetDrawingText()
        
        logger.debug(f"Generated atom coloring for molecule with {mol.GetNumAtoms()} atoms")
        return png
        
    except Exception as e:
        logger.error(f"Failed to color atoms on molecule: {e}")
        return None


def color_top_atoms(mol: Chem.Mol, 
                   scores: np.ndarray, 
                   top_ratio: float = 0.1, 
                   min_count: int = 1) -> Optional[bytes]:
    """
    Highlight only top important atoms based on score ranking.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        scores (np.ndarray): Atom importance scores (num_atoms,)
        top_ratio (float): Proportion of top atoms to highlight (0-1)
        min_count (int): Minimum number of atoms to highlight
        
    Returns:
        Optional[bytes]: PNG image bytes or None if failed
    """
    try:
        if len(scores) != mol.GetNumAtoms():
            raise ValueError(f"Score length ({len(scores)}) doesn't match number of atoms ({mol.GetNumAtoms()})")
        
        scores = np.asarray(scores, dtype=float)

        # Calculate number of atoms to select
        k = max(min_count, int(np.ceil(len(scores) * top_ratio)))
        k = min(k, len(scores))  # Ensure k doesn't exceed number of atoms
        
        if k == 0:
            logger.warning("No atoms selected for highlighting")
            return None

        # Find top k atoms
        kth = np.partition(scores, -k)[-k]  # k-th largest score
        sel = np.where(scores >= kth)[0].tolist()  # Selected atom indices

        if not sel:
            logger.warning("No atoms meet the selection criteria")
            return None

        # Normalize scores for selected atoms only
        s = scores[sel]
        s_min, s_max = s.min(), s.max()
        
        if s_max - s_min < 1e-8:
            norm = np.ones_like(s)  # Uniform coloring for equal scores
        else:
            norm = (s - s_min) / (s_max - s_min + 1e-8)

        # Create red gradient for selected atoms
        colors = {int(i): (1.0, 1 - float(v), 1 - float(v)) for i, v in zip(sel, norm)}

        # Generate molecular image
        d2d = Draw.MolDraw2DCairo(600, 400)
        d2d.DrawMolecule(
            mol,
            highlightAtoms=sel,
            highlightAtomColors=colors,
            highlightBonds=[]  # Don't highlight bonds
        )
        d2d.FinishDrawing()
        png = d2d.GetDrawingText()
        
        logger.debug(f"Highlighted top {len(sel)} atoms (top_ratio={top_ratio})")
        return png
        
    except Exception as e:
        logger.error(f"Failed to color top atoms: {e}")
        return None


def color_atoms_above_threshold(mol: Chem.Mol, 
                               scores: np.ndarray, 
                               threshold: float) -> Optional[bytes]:
    """
    Highlight atoms with importance scores above a specified threshold.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        scores (np.ndarray): Atom importance scores (num_atoms,)
        threshold (float): Score threshold for highlighting
        
    Returns:
        Optional[bytes]: PNG image bytes or None if failed
    """
    try:
        if len(scores) != mol.GetNumAtoms():
            raise ValueError(f"Score length ({len(scores)}) doesn't match number of atoms ({mol.GetNumAtoms()})")
        
        scores = np.asarray(scores, dtype=float)
        
        # Select atoms above threshold
        sel = np.where(scores > threshold)[0].tolist()

        if len(sel) == 0:
            logger.warning(f"No atoms above threshold {threshold}")
            return None

        # Normalize scores for selected atoms
        s = scores[sel]
        s_min, s_max = s.min(), s.max()
        
        if s_max - s_min < 1e-8:
            norm = np.ones_like(s)
        else:
            norm = (s - s_min) / (s_max - s_min + 1e-8)

        # Create light green to light blue gradient
        colors = {}
        for i, v in zip(sel, norm):
            # Light green (0.86, 0.93, 0.85) to light blue (0.50, 0.71, 0.84)
            r = 0.86 - 0.36 * float(v)
            g = 0.93 - 0.22 * float(v)
            b = 0.85 - 0.01 * float(v)
            colors[int(i)] = (r, g, b)

        # Generate molecular image
        d2d = Draw.MolDraw2DCairo(600, 400)
        d2d.DrawMolecule(
            mol,
            highlightAtoms=sel,
            highlightAtomColors=colors,
            highlightBonds=[]
        )
        d2d.FinishDrawing()
        png = d2d.GetDrawingText()
        
        logger.debug(f"Highlighted {len(sel)} atoms above threshold {threshold:.3f}")
        return png
        
    except Exception as e:
        logger.error(f"Failed to color atoms above threshold: {e}")
        return None


def color_atoms_by_score_segmented(mol: Chem.Mol, 
                                  scores: np.ndarray, 
                                  low_threshold: float = 0.3, 
                                  high_threshold: float = 0.8, 
                                  smooth: bool = False,
                                  color_scheme: str = 'pink') -> Optional[bytes]:
    """
    Color atoms using segmented coloring based on score ranges.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        scores (np.ndarray): Atom importance scores (num_atoms,)
        low_threshold (float): Lower threshold for medium importance
        high_threshold (float): Upper threshold for high importance
        smooth (bool): Whether to use smooth color transitions
        color_scheme (str): Color scheme ('pink' or 'green_blue')
        
    Returns:
        Optional[bytes]: PNG image bytes or None if failed
    """
    try:
        if len(scores) != mol.GetNumAtoms():
            raise ValueError(f"Score length ({len(scores)}) doesn't match number of atoms ({mol.GetNumAtoms()})")
        
        scores = np.asarray(scores, dtype=float)
        
        # Validate thresholds
        if low_threshold >= high_threshold:
            raise ValueError(f"low_threshold ({low_threshold}) must be less than high_threshold ({high_threshold})")
        
        # Select all atoms for coloring
        sel = list(range(len(scores)))

        # Define color schemes
        color_schemes = {
            'pink': {
                'light': (1.00, 0.9, 0.9),      # Very light pink
                'medium': (0.98, 0.75, 0.75),   # Medium pink
                'deep': (0.98, 0.55, 0.55)      # Deep pink
            },
            'green_blue': {
                'light': (0.8, 0.93, 0.75),     # Light green
                'medium': (0.87, 0.95, 0.97),   # Light blue-green
                'deep': (0.50, 0.71, 0.84)      # Deep blue-green
            }
        }
        
        if color_scheme not in color_schemes:
            logger.warning(f"Unknown color scheme '{color_scheme}', using 'pink'")
            color_scheme = 'pink'
            
        colors_config = color_schemes[color_scheme]
        light_color = colors_config['light']
        medium_color = colors_config['medium']
        deep_color = colors_config['deep']
        
        colors = {}
        for i in sel:
            score = scores[i]
            
            if smooth:
                # Smooth color transitions
                if score < low_threshold:
                    colors[int(i)] = light_color
                elif score < high_threshold:
                    t = (score - low_threshold) / (high_threshold - low_threshold)
                    r = light_color[0] + (medium_color[0] - light_color[0]) * t
                    g = light_color[1] + (medium_color[1] - light_color[1]) * t
                    b = light_color[2] + (medium_color[2] - light_color[2]) * t
                    colors[int(i)] = (r, g, b)
                else:
                    t = (score - high_threshold) / (1.0 - high_threshold)
                    r = medium_color[0] + (deep_color[0] - medium_color[0]) * t
                    g = medium_color[1] + (deep_color[1] - medium_color[1]) * t
                    b = medium_color[2] + (deep_color[2] - medium_color[2]) * t
                    colors[int(i)] = (r, g, b)
            else:
                # Hard segmentation
                if score < low_threshold:
                    colors[int(i)] = light_color
                elif score <= high_threshold:
                    colors[int(i)] = medium_color
                else:
                    colors[int(i)] = deep_color

        # Generate molecular image
        d2d = Draw.MolDraw2DCairo(600, 400)
        d2d.DrawMolecule(
            mol,
            highlightAtoms=sel,
            highlightAtomColors=colors,
            highlightBonds=[]
        )
        d2d.FinishDrawing()
        png = d2d.GetDrawingText()
        
        logger.debug(f"Applied segmented coloring ({color_scheme} scheme, smooth={smooth})")
        return png
        
    except Exception as e:
        logger.error(f"Failed to apply segmented coloring: {e}")
        return None


def save_visualization(png_data: bytes, 
                      filepath: str, 
                      overwrite: bool = False) -> bool:
    """
    Save molecular visualization to file.
    
    Args:
        png_data (bytes): PNG image data
        filepath (str): Path to save the image
        overwrite (bool): Whether to overwrite existing file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not png_data:
            logger.error("No image data to save")
            return False
            
        if os.path.exists(filepath) and not overwrite:
            logger.warning(f"File {filepath} already exists. Use overwrite=True to replace.")
            return False
            
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            f.write(png_data)
        
        logger.info(f"Visualization saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save visualization to {filepath}: {e}")
        return False


def explain_single_molecule(model: nn.Module,
                           smiles: str,
                           abeta_feature: torch.Tensor,
                           device: torch.device,
                           max_nodes: int = 75,
                           visualization_method: str = 'segmented',
                           output_dir: str = 'explain',
                           **kwargs) -> Dict[str, Any]:
    """
    Generate explanation for a single molecule.
    
    Args:
        model (nn.Module): Trained model
        smiles (str): SMILES string
        abeta_feature (torch.Tensor): Aβ protein features
        device (torch.device): Device for computation
        max_nodes (int): Maximum number of atoms
        visualization_method (str): Coloring method ('segmented', 'top', 'threshold', 'full')
        output_dir (str): Directory to save outputs
        **kwargs: Additional arguments for visualization functions
        
    Returns:
        Dict[str, Any]: Explanation results
    """
    try:
        from src.utils import generate_smile_graph, get_torch_data
        
        # Create molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Generate graph data
        smile_graph = generate_smile_graph([smiles], max_nodes)
        loader = get_torch_data([smiles], [0.0], smile_graph, batch_size=1, shuffle=False)
        
        # Get model attention
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            batch = batch.to(device)
            _, attn_dict = model(batch, abeta_feature, return_attn=True)
            att = attn_dict["fps_to_nodes"]  # (B, max_nodes)
            mask = attn_dict["node_mask"]    # (B, max_nodes)
        
        # Extract atom importance scores
        atom_importance = att[0, mask[0]].cpu().numpy()
        
        # Generate visualization based on method
        png_data = None
        if visualization_method == 'segmented':
            png_data = color_atoms_by_score_segmented(mol, atom_importance, **kwargs)
        elif visualization_method == 'top':
            png_data = color_top_atoms(mol, atom_importance, **kwargs)
        elif visualization_method == 'threshold':
            png_data = color_atoms_above_threshold(mol, atom_importance, **kwargs)
        elif visualization_method == 'full':
            png_data = color_atoms_on_mol(mol, atom_importance, **kwargs)
        else:
            logger.warning(f"Unknown visualization method: {visualization_method}, using 'segmented'")
            png_data = color_atoms_by_score_segmented(mol, atom_importance, **kwargs)
        
        # Save results
        results = {
            'smiles': smiles,
            'atom_importance': atom_importance,
            'num_atoms': mol.GetNumAtoms(),
            'visualization_method': visualization_method,
            'success': png_data is not None
        }
        
        if png_data:
            filename = f"explain_{visualization_method}_{hash(smiles) % 10000:04d}.png"
            filepath = os.path.join(output_dir, filename)
            save_visualization(png_data, filepath, overwrite=True)
            results['image_path'] = filepath
        
        logger.info(f"Generated explanation for {smiles} using {visualization_method} method")
        return results
        
    except Exception as e:
        logger.error(f"Failed to explain molecule {smiles}: {e}")
        return {
            'smiles': smiles,
            'error': str(e),
            'success': False
        }


def batch_explain_molecules(model: nn.Module,
                           smiles_list: List[str],
                           abeta_feature: torch.Tensor,
                           device: torch.device,
                           max_nodes: int = 75,
                           visualization_method: str = 'segmented',
                           output_dir: str = 'explain',
                           **kwargs) -> List[Dict[str, Any]]:
    """
    Generate explanations for multiple molecules.
    
    Args:
        model (nn.Module): Trained model
        smiles_list (List[str]): List of SMILES strings
        abeta_feature (torch.Tensor): Aβ protein features
        device (torch.device): Device for computation
        max_nodes (int): Maximum number of atoms
        visualization_method (str): Coloring method
        output_dir (str): Directory to save outputs
        **kwargs: Additional arguments for visualization
        
    Returns:
        List[Dict[str, Any]]: List of explanation results
    """
    results = []
    
    for i, smiles in enumerate(smiles_list):
        logger.info(f"Processing molecule {i+1}/{len(smiles_list)}: {smiles}")
        
        try:
            result = explain_single_molecule(
                model=model,
                smiles=smiles,
                abeta_feature=abeta_feature,
                device=device,
                max_nodes=max_nodes,
                visualization_method=visualization_method,
                output_dir=output_dir,
                **kwargs
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process molecule {i+1} ({smiles}): {e}")
            results.append({
                'smiles': smiles,
                'error': str(e),
                'success': False
            })
    
    # Generate summary report
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Batch explanation completed: {successful}/{len(results)} successful")
    
    return results


def create_explanation_report(results: List[Dict[str, Any]], 
                            output_file: str = 'explanation_report.csv') -> None:
    """
    Create a CSV report of explanation results.
    
    Args:
        results (List[Dict]): List of explanation results
        output_file (str): Output CSV file path
    """
    try:
        report_data = []
        for result in results:
            row = {
                'smiles': result.get('smiles', ''),
                'success': result.get('success', False),
                'num_atoms': result.get('num_atoms', 0),
                'visualization_method': result.get('visualization_method', ''),
                'image_path': result.get('image_path', ''),
                'error': result.get('error', '')
            }
            
            # Add atom importance statistics
            importance = result.get('atom_importance', [])
            if len(importance) > 0:
                row['importance_mean'] = np.mean(importance)
                row['importance_std'] = np.std(importance)
                row['importance_max'] = np.max(importance)
                row['importance_min'] = np.min(importance)
            else:
                row['importance_mean'] = row['importance_std'] = row['importance_max'] = row['importance_min'] = np.nan
                
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Explanation report saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to create explanation report: {e}")


# Convenience function for configuration-based explanation
def explain_from_config(config, 
                       model: nn.Module,
                       smiles_list: List[str],
                       abeta_feature: torch.Tensor,
                       output_dir: str = 'explain',
                       visualization_method: str = 'segmented',
                       **kwargs) -> List[Dict[str, Any]]:
    """
    Generate explanations using configuration parameters.
    
    Args:
        config: Configuration object
        model (nn.Module): Trained model
        smiles_list (List[str]): List of SMILES to explain
        abeta_feature (torch.Tensor): Aβ protein features
        output_dir (str): Output directory
        visualization_method (str): Visualization method
        **kwargs: Additional visualization parameters
        
    Returns:
        List[Dict[str, Any]]: Explanation results
    """
    logger.info(f"Starting batch explanation for {len(smiles_list)} molecules")
    
    results = batch_explain_molecules(
        model=model,
        smiles_list=smiles_list,
        abeta_feature=abeta_feature,
        device=config.device,
        max_nodes=config.max_nodes,
        visualization_method=visualization_method,
        output_dir=output_dir,
        **kwargs
    )
    
    # Save report
    report_path = os.path.join(output_dir, 'explanation_report.csv')
    create_explanation_report(results, report_path)
    
    logger.info(f"Explanation completed. Results saved to {output_dir}")
    return results