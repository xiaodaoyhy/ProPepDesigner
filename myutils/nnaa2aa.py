import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from typing import List, Tuple

from rdkit import RDLogger

# Completely disable all RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Dictionary of natural amino acid SMILES strings
natural_aa_smiles ={'R': 'N=C(N)NCCC[C@H](N)C(=O)O',
 'H': 'N[C@@H](Cc1c[nH]cn1)C(=O)O',
 'K': 'NCCCC[C@H](N)C(=O)O',
 'D': 'N[C@@H](CC(=O)O)C(=O)O',
 'E': 'N[C@@H](CCC(=O)O)C(=O)O',
 'S': 'N[C@@H](CO)C(=O)O',
 'T': 'C[C@@H](O)[C@H](N)C(=O)O',
 'N': 'NC(=O)C[C@H](N)C(=O)O',
 'Q': 'NC(=O)CC[C@H](N)C(=O)O',
 'C': 'N[C@@H](CS)C(=O)O',
 'G': 'NCC(=O)O',
 'A': 'C[C@H](N)C(=O)O',
 'P': 'O=C(O)[C@@H]1CCCN1',
 'I': 'CC[C@H](C)[C@H](N)C(=O)O',
 'L': 'CC(C)C[C@H](N)C(=O)O',
 'M': 'CSCC[C@H](N)C(=O)O',
 'F': 'N[C@@H](Cc1ccccc1)C(=O)O',
 'W': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
 'Y': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
 'V': 'CC(C)[C@H](N)C(=O)O'}


def calculate_all_descriptors(smiles_dict):
    """Calculate all available descriptors for each SMILES in the dictionary."""
    mols = {name: Chem.MolFromSmiles(smiles) for name, smiles in smiles_dict.items()}
    # Remove molecules that cannot be parsed
    mols = {name: mol for name, mol in mols.items() if mol is not None}
    # DataFrame to store all descriptor data
    all_results = []

    for name, mol in mols.items():
        # Use CalcMolDescriptors to calculate all descriptors
        desc_values = Descriptors.CalcMolDescriptors(mol)
        desc_values['Name'] = name  # Add the name column
        all_results.append(desc_values)

    # Create DataFrame
    df_all_desc = pd.DataFrame(all_results)
    df_all_desc.set_index('Name', inplace=True)
    
    return df_all_desc


def _prepare_natural_reference(method: str = "cosine"):
    """
    Precompute the natural amino acid descriptor reference matrix (computed only once).

    Returns:
        natural_names: pd.Index
        valid_cols: List[str]
        scaler: StandardScaler
        scaled_natural: np.ndarray, shape [n_natural, n_features]
    """
    if method not in {"euclidean", "cosine"}:
        raise ValueError("method must be 'euclidean' or 'cosine'")

    df_natural = calculate_all_descriptors(natural_aa_smiles)

    # Remove columns containing NaN/Inf (natural reference side)
    valid_cols = []
    for col in df_natural.columns:
        if df_natural[col].isna().any() or np.isinf(df_natural[col]).any():
            continue
        valid_cols.append(col)

    df_natural = df_natural[valid_cols]

    scaler = StandardScaler()
    scaled_natural = scaler.fit_transform(df_natural.values)
    natural_names = df_natural.index

    return natural_names, valid_cols, scaler, scaled_natural


# Pre-prepare the natural AA reference at module load time (avoid recomputing on every call)
_NATURAL_NAMES, _VALID_COLS, _SCALER, _SCALED_NATURAL = _prepare_natural_reference(method="cosine")

def map_single_smiles_to_aa(smiles: str, topk: int = 1):
    """
    Input the SMILES of a single amino acid, return the single-letter code of the most similar natural amino acid.

    Notes:
    - RDKit's CalcMolDescriptors is used to generate a set of molecular descriptors,
      which are then matched for similarity against the "natural amino acid (monomer)" reference set.
    - For efficiency, the natural amino acid reference descriptors are precomputed once at module load time.

    Args:
        smiles: str
            SMILES of the non-natural amino acid

        topk: int
            Return topk candidates (with scores) so you can check whether the mapping is reasonable

    Returns:
        best_match: str
            The single-letter code of the best-matching natural amino acid
        topk_matches: List[Tuple[str, float]]
            List of topk candidates [(aa, score), ...]
            - When method='cosine', score is the similarity (higher is better)
    """

    mol = Chem.MolFromSmiles(smiles)

    # Compute the query descriptors and align them to valid_cols
    desc = Descriptors.CalcMolDescriptors(mol)
    x = np.array([desc[c] for c in _VALID_COLS], dtype=float).reshape(1, -1)

    # Standardize to the natural AA distribution
    x_scaled = _SCALER.transform(x)[0]

    scores: List[Tuple[str, float]] = []
    for j, aa in enumerate(_NATURAL_NAMES):
        nat_vec = _SCALED_NATURAL[j]
        score = float(1.0 - cosine(x_scaled, nat_vec))
        scores.append((aa, score))


    scores.sort(key=lambda t: t[1], reverse=True)  # Higher similarity first
    best_match = scores[0][0]
    return best_match, scores[: max(1, int(topk))]




if __name__ == "__main__":

    test_smiles = "N[C@](CC(C)C)(C)C(=O)O"  # SMILES of the V monomer
    best, top = map_single_smiles_to_aa(test_smiles, topk=1)
    best = map_single_smiles_to_aa(test_smiles, topk=1)

    print("Best:", best)
    print("Top3:", top)
