# -*- coding: utf-8 -*-
"""
Utility functions for molecular graph processing, feature extraction, and data handling.

Author: Your Name
Date: August 2024
"""

import random
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import networkx as nx
from torch_geometric import data as DATA
from torch_geometric.data import DataLoader
from tqdm import tqdm
import warnings
from typing import List, Tuple, Dict, Any, Union
import os

warnings.filterwarnings('ignore')


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    Extract comprehensive features for a single atom.
    
    Args:
        atom (Chem.Atom): RDKit atom object
        
    Returns:
        np.ndarray: 43-dimensional atom feature vector
    """
    try:
        features = np.array(
            one_of_k_encoding_unk(atom.GetSymbol(), 
                                ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Se']) +
            one_of_k_encoding(atom.GetDegree(), [1, 2, 3, 4]) +
            one_of_k_encoding(atom.GetTotalDegree(), [1, 2, 3, 4]) +
            one_of_k_encoding_unk(atom.GetNumImplicitHs(), [0, 1, 2, 3]) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3]) +                    
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) +
            one_of_k_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6]) +
            one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) +
            one_of_k_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3']) +
            [atom.GetIsAromatic()]
        )
        return features
    except Exception as e:
        raise ValueError(f"Failed to extract features for atom: {e}")


def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[bool]:
    """
    One-hot encoding for known values.
    
    Args:
        x: Value to encode
        allowable_set: List of allowable values
        
    Returns:
        List[bool]: One-hot encoded vector
        
    Raises:
        ValueError: If x is not in allowable_set
    """
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: Any, allowable_set: List[Any]) -> List[bool]:
    """
    One-hot encoding with unknown category for values not in allowable set.
    
    Args:
        x: Value to encode
        allowable_set: List of allowable values
        
    Returns:
        List[bool]: One-hot encoded vector with unknown category
    """
    if x not in allowable_set:
        x = allowable_set[-1]  # Map to unknown category
    return list(map(lambda s: x == s, allowable_set))


def smiles_to_padded_graph(smile: str, max_nodes: int) -> Tuple[int, torch.Tensor, np.ndarray]:
    """
    Convert SMILES string to padded graph representation.
    
    Args:
        smile (str): SMILES string
        max_nodes (int): Maximum number of nodes for padding
        
    Returns:
        Tuple[int, torch.Tensor, np.ndarray]: 
            - Number of atoms
            - Padded node features (max_nodes, feature_dim)
            - Edge index array
            
    Raises:
        ValueError: If SMILES is invalid or molecule exceeds max_nodes
    """
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smile}")
            
        # Handle salts by removing metal ions
        if is_salt(mol):
            mol = remove_metal(mol)
            if mol is None:
                raise ValueError(f"Failed to process salt molecule: {smile}")
        
        # Extract atom features
        features_list = []
        for atom in mol.GetAtoms():
            features = atom_features(atom)
            features_list.append(features)
        
        features = torch.tensor(features_list, dtype=torch.float)  # (num_atoms, 43)
        
        # Padding to max_nodes
        num_atoms = features.size(0)
        feature_dim = features.size(1)
        
        if num_atoms > max_nodes:
            raise ValueError(f"Molecule has {num_atoms} atoms, exceeds max_nodes {max_nodes}. "
                           f"Consider increasing max_nodes or filtering large molecules.")
        
        if num_atoms < max_nodes:
            padding = torch.zeros((max_nodes - num_atoms, feature_dim))
            features = torch.cat([features, padding], dim=0)  # Pad to (max_nodes, feature_dim)

        # Extract bond information
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        
        # Create directed graph
        g = nx.Graph(edges).to_directed()
        
        # Convert to edge index
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        
        edge_index = np.array(edge_index, dtype=np.int64)
        
        return num_atoms, features, edge_index
        
    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to graph: {smile}. Error: {e}")


def generate_smile_graph(mols: List[str], max_nodes: int) -> Dict[str, Tuple]:
    """
    Generate graph representations for a list of SMILES strings.
    
    Args:
        mols (List[str]): List of SMILES strings
        max_nodes (int): Maximum number of nodes for padding
        
    Returns:
        Dict[str, Tuple]: Dictionary mapping SMILES to graph tuples
        
    Raises:
        ValueError: If any SMILES conversion fails
    """
    smile_graph = {}
    failed_smiles = []
    
    for smile in tqdm(mols, desc="Generating molecular graphs"):
        try:
            graph_data = smiles_to_padded_graph(smile, max_nodes)
            smile_graph[smile] = graph_data
        except Exception as e:
            failed_smiles.append((smile, str(e)))
    
    if failed_smiles:
        print(f"Warning: Failed to process {len(failed_smiles)} SMILES strings")
        for smile, error in failed_smiles[:5]:  # Show first 5 errors
            print(f"  {smile}: {error}")
    
    return smile_graph


def smiles_to_ecfp(smiles: Union[str, List[str]], 
                   radius: int = 2, 
                   nbits: int = 2048, 
                   to_array: bool = True) -> Union[List, np.ndarray]:
    """
    Generate ECFP fingerprints from SMILES strings.
    
    Args:
        smiles (Union[str, List[str]]): SMILES string or list of SMILES
        radius (int): Morgan fingerprint radius
        nbits (int): Fingerprint bit length
        to_array (bool): Whether to convert to numpy array
        
    Returns:
        Union[List, np.ndarray]: ECFP fingerprints
        
    Raises:
        ValueError: If SMILES conversion fails
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    
    fp_list = []
    failed_smiles = []
    
    for s in smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {s}")
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fp_list.append(fp)
        except Exception as e:
            failed_smiles.append((s, str(e)))
            # Add zero vector for failed conversions
            fp_list.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("C"), radius, nBits=nbits))
    
    if failed_smiles:
        print(f"Warning: Failed to generate ECFP for {len(failed_smiles)} SMILES")
    
    if not to_array:
        return fp_list
    
    # Convert to numpy array
    output = []
    for f in fp_list:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)
        
    return np.asarray(output)


def get_torch_data(smiles: List[str], 
                   labels: List[float], 
                   smile_graph: Dict[str, Tuple], 
                   batch_size: int, 
                   shuffle: bool) -> DataLoader:
    """
    Create PyTorch Geometric DataLoader for training/validation.
    
    Args:
        smiles (List[str]): List of SMILES strings
        labels (List[float]): List of target values
        smile_graph (Dict): Precomputed graph dictionary
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: PyTorch Geometric DataLoader
        
    Raises:
        ValueError: If inputs have inconsistent lengths
    """
    if len(smiles) != len(labels):
        raise ValueError(f"SMILES length ({len(smiles)}) != labels length ({len(labels)})")
    
    try:
        ecfp_df = smiles_to_ecfp(smiles)
    except Exception as e:
        raise ValueError(f"Failed to generate ECFP fingerprints: {e}")
    
    torch_data = []
    
    for i in tqdm(range(len(smiles)), desc="Creating PyTorch data"):
        try:
            smile = smiles[i]
            label = labels[i]
            ecfp = ecfp_df[i]
            
            if smile not in smile_graph:
                raise KeyError(f"SMILES {smile} not found in precomputed graphs")
                
            c_size, features, edge_index = smile_graph[smile]
            
            # Create PyG Data object
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([label])
            )
            GCNData.fps = torch.Tensor([ecfp])
            
            torch_data.append(GCNData)
            
        except Exception as e:
            print(f"Warning: Failed to process molecule {i} ({smiles[i]}): {e}")
            continue
    
    if not torch_data:
        raise ValueError("No valid data objects created")
    
    return DataLoader(torch_data, batch_size=batch_size, shuffle=shuffle)


def get_torch_data_unlabel(smiles: List[str], 
                          smile_graph: Dict[str, Tuple], 
                          batch_size: int, 
                          shuffle: bool) -> DataLoader:
    """
    Create PyTorch Geometric DataLoader for unlabeled data.
    
    Args:
        smiles (List[str]): List of SMILES strings
        smile_graph (Dict): Precomputed graph dictionary
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: PyTorch Geometric DataLoader
    """
    try:
        ecfp_df = smiles_to_ecfp(smiles)
    except Exception as e:
        raise ValueError(f"Failed to generate ECFP fingerprints: {e}")
    
    torch_data = []
    
    for i in range(len(smiles)):
        try:
            smile = smiles[i]
            ecfp = ecfp_df[i]
            
            if smile not in smile_graph:
                raise KeyError(f"SMILES {smile} not found in precomputed graphs")
                
            c_size, features, edge_index = smile_graph[smile]
            
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0)
            )
            GCNData.fps = torch.Tensor([ecfp])
            
            torch_data.append(GCNData)
            
        except Exception as e:
            print(f"Warning: Failed to process unlabeled molecule {i}: {e}")
            continue
    
    if not torch_data:
        raise ValueError("No valid unlabeled data objects created")
    
    return DataLoader(torch_data, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def check_featurizability(smiles: str) -> bool:
    """
    Check if a SMILES string can be successfully featurized.
    
    Args:
        smiles (str): SMILES string to check
        
    Returns:
        bool: True if featurizable, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return False
            
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for atom in mol.GetAtoms():
            try:
                _ = atom_features(atom)
            except:
                return False
        return True
    except:
        return False


def is_salt(mol: Chem.Mol) -> bool:
    """
    Check if molecule contains metal ions (is a salt).
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        bool: True if molecule contains metal ions
    """
    metal_ions_list = [
        '[Li', '[Be', '[Na', '[Mg', '[Al', '[K', '[Ca', '[Sc', '[Ti', '[V', '[Cr', '[Mn', '[Fe', '[Co',
        '[Ni', '[Cu', '[Zn', '[Ga', '[Rb', '[Sr', '[Y', '[Zr', '[Nb', '[Mo', '[Tc', '[Ru', '[Rh', '[Pd',
        '[Ag', '[Cd', '[In', '[Sn', '[Cs', '[Ba', '[La', '[Ce', '[Pr', '[Nd', '[Pm', '[Sm', '[Eu', '[Gd',
        '[Tb', '[Dy', '[Ho', '[Er', '[Tm', '[Yb', '[Lu', '[Hf', '[Ta', '[W', '[Re', '[Os', '[Ir', '[Pt',
        '[Au', '[Hg', '[Tl', '[Pb', '[Bi', '[Po', '[Fr', '[Ra', '[Ac', '[Th', '[Pa', '[U', '[Np', '[Pu',
        '[Am', '[Cm', '[Bk', '[Cf', '[Es', '[Fm', '[Md', '[No', '[Lr', '[Rf', '[Db', '[Sg', '[Bh', '[Hs',
        '[Mt', '[Ds', '[Rg', '[Cn', '[Nh', '[Fl', '[Mc', '[Lv]'
    ]
    
    try:
        smi = Chem.MolToSmiles(mol)
        for metal_ion in metal_ions_list:
            if metal_ion in smi:
                return True
        return False
    except:
        return False


def remove_metal(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove metal ions and atoms from molecule.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        Chem.Mol: Molecule with metal ions removed
    """
    metal_list = [
        'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
        'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra',
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
        'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv'
    ]
    
    try:
        mol_rm = Chem.RWMol(mol)
        atoms = mol.GetAtoms()
        
        # Remove metal atoms in reverse order to avoid index issues
        metal_indices = []
        for atom in atoms:
            if atom.GetSymbol() in metal_list:
                metal_indices.append(atom.GetIdx())
        
        # Remove from highest index to lowest
        for idx in sorted(metal_indices, reverse=True):
            mol_rm.RemoveAtom(idx)
        
        # Convert back to molecule
        if mol_rm.GetNumAtoms() > 0:
            smi = Chem.MolToSmiles(mol_rm)
            return Chem.MolFromSmiles(smi)
        else:
            return None
    except Exception as e:
        print(f"Warning: Failed to remove metals from molecule: {e}")
        return mol


class ZeroScoreNorm:
    """
    Zero-score normalization (standardization) for data preprocessing.
    """
    
    def __init__(self, values: np.ndarray):
        """
        Initialize normalizer with data statistics.
        
        Args:
            values (np.ndarray): Data values for computing statistics
        """
        self.avg = np.average(values)
        self.std = np.std(values)
        print(f"Normalizer initialized: mean={self.avg:.4f}, std={self.std:.4f}")

    def norm(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values using z-score.
        
        Args:
            values (np.ndarray): Values to normalize
            
        Returns:
            np.ndarray: Normalized values
        """
        values = np.array(values, dtype=float)
        return (values - self.avg) / (self.std + 1e-8)

    def recovery(self, values: np.ndarray) -> np.ndarray:
        """
        Recover original values from normalized values.
        
        Args:
            values (np.ndarray): Normalized values
            
        Returns:
            np.ndarray: Original scale values
        """
        values = np.array(values, dtype=float)
        return values * self.std + self.avg


def compute_aac(sequence: str) -> np.ndarray:
    """
    Compute Amino Acid Composition (AAC) feature vector.
    
    Args:
        sequence (str): Protein amino acid sequence
        
    Returns:
        np.ndarray: 20-dimensional AAC feature vector
        
    Raises:
        ValueError: If sequence is empty or contains invalid amino acids
    """
    if not sequence:
        raise ValueError("Empty protein sequence")
    
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq_length = len(sequence)
    
    # Validate sequence
    invalid_aa = set(sequence) - set(amino_acids)
    if invalid_aa:
        raise ValueError(f"Sequence contains invalid amino acids: {invalid_aa}")
    
    # Compute amino acid composition
    encoding = np.array([sequence.count(aa) / seq_length for aa in amino_acids])
    
    return encoding


def safe_file_operation(filepath: str, mode: str = 'r'):
    """
    Context manager for safe file operations with error handling.
    
    Args:
        filepath (str): Path to file
        mode (str): File opening mode
        
    Yields:
        file object
        
    Raises:
        IOError: If file operation fails
    """
    try:
        with open(filepath, mode) as f:
            yield f
    except Exception as e:
        raise IOError(f"Failed to operate on file {filepath}: {e}")


# Convenience function for configuration-based data loading
def load_data_from_config(config, data_type: str = 'train') -> DataLoader:
    """
    Load data using configuration parameters.
    
    Args:
        config: Configuration object
        data_type (str): Type of data to load ('train', 'val', 'test')
        
    Returns:
        DataLoader: PyTorch Geometric DataLoader
    """
    if data_type == 'train':
        filepath = os.path.join(config.data_dir, config.train_file)
    elif data_type == 'val':
        filepath = os.path.join(config.data_dir, config.val_file)
    elif data_type == 'test':
        filepath = os.path.join(config.data_dir, config.test_file)
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    try:
        data = pd.read_csv(filepath)
        smiles = data.iloc[:, 1].tolist()
        labels = data.iloc[:, 2].tolist()
        
        graph = generate_smile_graph(smiles, config.max_nodes)
        loader = get_torch_data(smiles, labels, graph, config.batch_size, shuffle=(data_type == 'train'))
        
        return loader
    except Exception as e:
        raise ValueError(f"Failed to load {data_type} data from {filepath}: {e}")