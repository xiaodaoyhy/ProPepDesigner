import torch
from torch.utils.data import TensorDataset
from typing import Optional, List, Union
import numpy as np
import joblib
from pathlib import Path
from joblib import Parallel, delayed

import esm
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.*')
from myutils.nnaa2aa import map_single_smiles_to_aa
from propy.PyPro import GetProDes


NAA2INDEX = {'Aib': 0, 'aMeL': 1, 'Orn': 2, 
             'aMeFPhe': 3, 'aMeTyr': 4, 'dE': 5, 
             'aMe4Pal': 6, 'aMeSer': 7, 
             'dA': 8, 'aMePhe': 9, '4Pal':10, 'X':11
}


def count_nnaa_presence(fasta_list: List[List[str]]) -> torch.Tensor:
    vocab_size = len(NAA2INDEX)
    batch_vec = torch.zeros((len(fasta_list), vocab_size), dtype=torch.float32)
    AA_STD = set("ACDEFGHIKLMNPQRSTVWY")
    for b, seq in enumerate(fasta_list):
        present = { (tok.replace('*','') if tok.replace('*','') in NAA2INDEX else 'X') 
                    for tok in seq if tok.replace('*','') not in AA_STD and tok.replace('*','') }
        for tok in present:
            batch_vec[b, NAA2INDEX[tok]] = 1.0
    # Return whether each amino acid in NAA2INDEX is present
    return batch_vec # [batch_size, vocab_size]


def calc_single_ecfp4(smiles: str, radius: int = 2, nBits: int = 1024) -> np.ndarray:
    """Compute ECFP4 fingerprint (Morgan fingerprint) for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    # radius=2 -> ECFP4, radius=3 -> ECFP6
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



def calc_single_rdkit(smiles: str) -> List[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [0.0] * 14
    return [
            Descriptors.ExactMolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.NumHeteroatoms(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),  # Number of fully saturated rings
            Descriptors.NumValenceElectrons(mol),
    ]



def calc_single_physchem(clean_seq_list: List[str]) -> np.ndarray:
    clean_seq_str = "".join([c if c in "ACDEFGHIKLMNPQRSTVWY" else "A" for c in clean_seq_list])
    pa = ProteinAnalysis(clean_seq_str)
    ss = pa.secondary_structure_fraction()

    return np.array([
        float(len(clean_seq_list)), pa.isoelectric_point(),
        pa.aromaticity(), pa.instability_index(), 
        np.mean(pa.flexibility()) if clean_seq_str else 0.0,
        pa.gravy(), pa.charge_at_pH(7.4), ss[0], ss[1], ss[2]
    ], dtype=np.float32)


def calc_single_ctd(clean_seq_list: List[str]) -> np.ndarray:
    """Compute Propy3 CTD descriptors (147-dim)."""
    # Convert to Propy-compatible single-letter string; unknown residues -> A
    seq_str = "".join([c if c in "ACDEFGHIKLMNPQRSTVWY" else "A" for c in clean_seq_list])
    
    if not seq_str:
        return np.zeros(147, dtype=np.float32)
        
    try:
        des = GetProDes(seq_str)
        ctd_dict = des.GetCTD()
        return np.array(list(ctd_dict.values()), dtype=np.float32)
    except:
        return np.zeros(147, dtype=np.float32)



def create_esm_embedding(clean_fasta_ls: List[List[str]], device="cuda") -> torch.Tensor:
    """Batch-compute ESM embeddings."""
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    
    # Convert to standard string sequences
    sequences = ["".join([c if len(c)==1 and c in "ACDEFGHIKLMNPQRSTVWY" else "X" for c in seq]) 
                 for seq in clean_fasta_ls]
    
    esm_embeddings_mean = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            labels, strs, tokens = batch_converter([(str(j), s) for j, s in enumerate(batch_seqs)])
            tokens = tokens.to(device)
            
            results = model(tokens, repr_layers=[model.num_layers])
            token_representations = results["representations"][model.num_layers]
            
            # Mean pooling (exclude BOS/EOS tokens)
            for j, seq in enumerate(batch_seqs):
                mean_emb = token_representations[j, 1 : len(seq) + 1].mean(0).cpu()
                esm_embeddings_mean.append(mean_emb)
                
    return torch.stack(esm_embeddings_mean)



def sanitize_sequence(seq: List[str], smiles: List[str]) -> List[str]:
    AA_STD_SET = set("ACDEFGHIKLMNPQRSTVWY")
    new_seq = []
    for aa, smile in zip(seq, smiles):
        temp_aa = aa.replace('*', '')
        if temp_aa.startswith('d') and len(temp_aa) > 1:
            temp_aa = temp_aa[1:]
        if temp_aa in AA_STD_SET:
            new_seq.append(temp_aa)
        else:
            mapped = map_single_smiles_to_aa(smile, topk=1)[0]
            new_seq.append(mapped if (isinstance(mapped, str) and mapped in AA_STD_SET) else 'X')
    return new_seq



# --- Main function ---

def getEmbedding(
    fasta_list: List[List[str]],
    frag_list: List[List[str]],
    main_smiles_list: List[str],
    feature_type: List[str],
    run_type: str = 'train',
    activity_array: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> TensorDataset:
    
    # Feature tensor placeholders
    physchem_tensor = None
    rdkit_tensor = None
    ctd_tensor = None  
    ecfp4_tensor = None
    esm_tensor = None
    nnaa_tensor = None

    # AA mapping (CPU parallel)
    clean_fasta_ls = Parallel(n_jobs=4)(delayed(sanitize_sequence)(f, s) for f, s in zip(fasta_list, frag_list))

    # Feature computation
    if 'PhysChem' in feature_type:
        print("Computing PhysChem features...")
        physchem_feats = Parallel(n_jobs=4)(delayed(calc_single_physchem)(f) for f in clean_fasta_ls)
        physchem_tensor = torch.tensor(np.array(physchem_feats), dtype=torch.float32)
    
    if 'RDKit' in feature_type:
        print("Computing RDKit features...")
        rdkit_feats = Parallel(n_jobs=4)(delayed(calc_single_rdkit)(s) for s in main_smiles_list)
        rdkit_tensor = torch.tensor(np.array(rdkit_feats), dtype=torch.float32)

    if 'CTD' in feature_type: 
        print("Computing Propy3 CTD features...")
        ctd_feats = Parallel(n_jobs=4)(delayed(calc_single_ctd)(f) for f in clean_fasta_ls)
        ctd_tensor = torch.tensor(np.array(ctd_feats), dtype=torch.float32)

    if 'ECFP4' in feature_type:
        print("Computing ECFP4 fingerprints...")
        ecfp4_feats = Parallel(n_jobs=4)(delayed(calc_single_ecfp4)(s, nBits=1024) for s in main_smiles_list)
        ecfp4_tensor = torch.tensor(np.array(ecfp4_feats), dtype=torch.float32)
    
    if 'ESM' in feature_type:
        print("Computing ESM embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        esm_tensor = create_esm_embedding(clean_fasta_ls, device=device)
    
    if 'NNAA' in feature_type:
        print("Computing NNAA features...")
        nnaa_tensor = count_nnaa_presence(fasta_list)
    
    # --- Standardization ---
    # Standardize CTD together with RDKit/PhysChem features
    numeric_parts = []
    if rdkit_tensor is not None: numeric_parts.append(rdkit_tensor)
    if physchem_tensor is not None: numeric_parts.append(physchem_tensor)
    if ctd_tensor is not None: numeric_parts.append(ctd_tensor)

    if len(numeric_parts) > 0:
        raw_numeric_feats = torch.cat(numeric_parts, dim=1)
        
        save_path_obj = Path(save_path) if save_path else Path('.')
        save_scaler_path = save_path_obj / 'feature_scaler.pkl'
        
        raw_feats_np = raw_numeric_feats.numpy()
        if run_type == 'train':
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(raw_feats_np)
  
            joblib.dump(scaler, save_scaler_path)

        else:
            scaler = joblib.load(save_scaler_path)
            numeric_scaled = scaler.transform(raw_feats_np)
        numeric_scaled_tensor = torch.as_tensor(numeric_scaled, dtype=torch.float32)

    else:
        numeric_scaled_tensor = torch.zeros(len(fasta_list), 0)

    # --- Final concatenation by requested features ---
    parts = [numeric_scaled_tensor]
    if 'ESM' in feature_type and esm_tensor is not None:
        parts.append(esm_tensor)
    if 'ECFP4' in feature_type and ecfp4_tensor is not None:
        parts.append(ecfp4_tensor)
    if 'NNAA' in feature_type and nnaa_tensor is not None:
        parts.append(nnaa_tensor)
    
    emb_x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    print(f'Feature dimensions: {emb_x.shape}')
    
    # Label handling
    if activity_array is not None:
        activity_np = np.asarray(activity_array)
        if activity_np.ndim == 1:
            activity_np = activity_np.reshape(-1, 1)
        activity_tensor = torch.as_tensor(activity_np, dtype=torch.float32)
        activity_mask_tensor = torch.as_tensor((~np.isnan(activity_np)).astype(np.float32))
    else:
        activity_tensor = torch.full((emb_x.shape[0], 3), float('nan'))
        activity_mask_tensor = torch.zeros((emb_x.shape[0], 3))
    
    return TensorDataset(emb_x, activity_tensor, activity_mask_tensor)