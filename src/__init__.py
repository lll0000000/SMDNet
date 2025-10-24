# src/__init__.py
"""
SMDNet - Interpretable Molecular Property Prediction
"""

__version__ = "1.0.0"
__author__ = "Yanling Wu"

from .model import CrossAttGATNet_AdaLN
from .utils import (
    generate_smile_graph, 
    get_torch_data, 
    get_torch_data_unlabel,
    set_seed, 
    compute_aac
)
from .train import train_and_val, test
from .explain import (
    color_atoms_on_mol,
    color_top_atoms, 
    color_atoms_above_threshold,
    color_atoms_by_score_segmented
)
from .screening import (
    mc_dropout_predictions,
    compute_mutual_information,
    candidate_selection,
    reverse_sampling_with_mi
)

__all__ = [
    "CrossAttGATNet_AdaLN",
    "generate_smile_graph",
    "get_torch_data", 
    "get_torch_data_unlabel",
    "set_seed",
    "compute_aac",
    "train_and_val", 
    "test",
    "color_atoms_on_mol",
    "color_top_atoms",
    "color_atoms_above_threshold", 
    "color_atoms_by_score_segmented",
    "mc_dropout_predictions",
    "compute_mutual_information",
    "candidate_selection",
    "reverse_sampling_with_mi"
]