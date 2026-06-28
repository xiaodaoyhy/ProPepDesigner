import torch
from torch.utils.data import Dataset
from typing import Optional, List, Union
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from pathlib import Path
import esm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from myutils.nnaa2aa import map_single_smiles_to_aa


class EmbeddingDataset(Dataset):
    def __init__(
        self, 
        embeddings: List[torch.Tensor], 
        fasta_list: List[List],
        smiles_list: List[List[str]],
        max_len: int = 45, 
        targets: Optional[torch.Tensor] = None
    ):
        """
        Initialize dataset
        
        Args:
            org_mol: Original SMILES list
            embeddings: List of embedding tensors, each shaped [seq_len, d_model]
            max_len: Maximum sequence length
            targets: Optional target tensor, shaped [num_samples]
        """
        self.embeddings = self._pad_sequences(embeddings, max_len)  # Pad/truncate embeddings
        
        seq_lens = torch.tensor([emb.size(0) for emb in embeddings])
        self.masks = torch.arange(max_len).expand(len(embeddings), max_len) >= seq_lens.unsqueeze(1)
        self.fasta_list = fasta_list
        self.smiles_list = smiles_list
        
        if targets is not None:
            assert len(self.embeddings) == len(targets), "Embedding and target lengths do not match"
        self.targets = torch.tensor(targets, dtype=torch.float) if targets is not None else None
        
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        """
        Return one sample for DataLoader:
        - train/val/test: (embeddings, masks, targets)
        - predict: (embeddings, masks)
        """
        if self.targets is not None:
            return self.embeddings[idx], self.masks[idx], self.targets[idx]
        else:
            return self.embeddings[idx], self.masks[idx]
    
    @staticmethod
    def _pad_sequences(embeddings: List[torch.Tensor], max_len: int) -> torch.Tensor:
        """
        Pad or truncate sequences to fixed length
        
        Args:
            embeddings: List of embedding tensors, each shaped [seq_len, d_model]
            max_len: Target sequence length
            
        Returns:
            Tensor shaped [num_samples, max_len, d_model]
        """
        device = embeddings[0].device
        
        batch_size = len(embeddings)
        d_model = embeddings[0].size(1)
        padded = torch.zeros((batch_size, max_len, d_model), device=device)
        
        for i, emb in enumerate(embeddings):
            seq_len = min(emb.size(0), max_len)
            padded[i, :seq_len] = emb[:seq_len]
            
        return padded




def create_onehot_embedding(fasta_list):
    # Amino acids with frequency > 0.1%
    AA2INDEX = {'S': 0, 'G': 1, 'P': 2, 'E': 3, 'L': 4, 'A': 5, 'T': 6, 'D': 7, \
                'I': 8, 'K': 9, 'Q': 10, 'F': 11, 'Y': 12, 'Aib': 13, 'aMeL': 14,\
                'Orn': 15, 'W': 16, 'V': 17, 'R': 18, 'H': 19, '4Pal': 20, 'aMeFPhe': 21,\
                'aMeTyr': 22, 'N': 23, 'dE': 24, 'aMe4Pal': 25, 'aMeSer': 26, 'M': 27, 'dA': 28,\
                'C': 29,  'X': 30}


    # Build one-hot tensors
    onehot_embeddings = []
    for fasta in fasta_list:
        # Build on CPU; move to GPU per batch during training
        # One-hot dim matches AA2INDEX size (includes 'X' for unknown residues)
        embedding = torch.zeros(len(fasta), len(AA2INDEX.keys()), dtype=torch.float32)
        for i, aa in enumerate(fasta):
            aa = aa.replace('*', '')
            # Determine amino acid category
            current_aa = aa if aa in AA2INDEX.keys() else 'X'
            embedding[i, AA2INDEX[current_aa]] = 1.0
        onehot_embeddings.append(embedding)
    return onehot_embeddings





def create_rdkit_embedding(smiles_list, save_scaler_path, scaler=None, device="cpu"):
    """
    Build RDKit physicochemical descriptor embeddings
    - Training: scaler=None, save_scaler_path="xxx.pkl" -> fit and save automatically
    - Prediction: scaler=loaded scaler object -> transform directly
    """
    # Collect all fragments and record original positions
    all_fragments = []
    fragment_indices = []

    for smiles in smiles_list:
        start_idx = len(all_fragments)
        fragments = list(smiles)
        all_fragments.extend(fragments)
        fragment_indices.append((start_idx, len(all_fragments)))

    unique_fragments = []
    fragment_to_unique = {}
    unique_to_original = []

    for idx, frag in enumerate(all_fragments):
        if frag not in fragment_to_unique:
            fragment_to_unique[frag] = len(unique_fragments)
            unique_fragments.append(frag)
            unique_to_original.append([idx])
        else:
            unique_to_original[fragment_to_unique[frag]].append(idx)

    # Descriptor computation
    def calc_descriptors(mol):
        desc = [
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
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            Descriptors.NumValenceElectrons(mol),
        ]


        # print(desc)
        return np.array(desc, dtype=np.float32)

    # Compute descriptors for unique fragments
    unique_descriptors = []
    for frag in unique_fragments:
        mol = Chem.MolFromSmiles(frag)
        assert mol is not None, f"SMILES error: {frag}"
        desc = calc_descriptors(mol)
        unique_descriptors.append(desc)

    unique_descriptors = np.stack(unique_descriptors)  # [num_unique_fragments, num_descriptors]

    # Normalization (StandardScaler)
    if scaler is None:
        # Training phase
        scaler = StandardScaler()
        unique_descriptors = scaler.fit_transform(unique_descriptors)
        joblib.dump(scaler, save_scaler_path)

    else:
        # Prediction phase
        unique_descriptors = scaler.transform(unique_descriptors)

    # Map back to all fragments
    all_embeddings = [None] * len(all_fragments)
    for unique_idx, original_indices in enumerate(unique_to_original):
        # Keep on CPU (or specified device) to avoid high GPU memory use during embedding
        desc_tensor = torch.from_numpy(unique_descriptors[unique_idx]).to(device)
        for orig_idx in original_indices:
            all_embeddings[orig_idx] = desc_tensor

    # Group by peptide
    descriptor_embeddings = []
    for start, end in fragment_indices:
        descriptor_embeddings.append(torch.stack(all_embeddings[start:end]))

    return descriptor_embeddings




def create_esm_embedding(
    fasta_list: List[List],
    device: Optional[Union[str, torch.device]] = None,
) -> List[torch.Tensor]:

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load pretrained model ('esm2_t12_35M_UR50D': 480, 'esm2_t6_8M_UR50D': 320)
    model, alphabet = getattr(esm.pretrained, 'esm2_t6_8M_UR50D')() 
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()


    # For ESM2, use last-layer representations by default
    repr_layer = model.num_layers

    CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def tokens_to_seq(tokens: List[str]) -> str:
        seq_chars = []
        for tok in tokens:
            if len(tok) == 1 and tok in CANONICAL_AA:
                seq_chars.append(tok)
            else:
                # Map non-standard residues to 'X'
                seq_chars.append("X")
        return "".join(seq_chars)

    sequences = [tokens_to_seq(tokens) for tokens in fasta_list]

    esm_embeddings: List[torch.Tensor] = []
    batch_size = 32  # ESM forward batch size; adjust based on GPU memory

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            batch_labels = [(str(i), seq) for i, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(batch_labels)
            batch_tokens = batch_tokens.to(device)

            out = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_reprs = out["representations"][repr_layer]  # [B, L+2, d_esm]

            for i, seq in enumerate(batch_seqs):
                # Remove BOS/EOS; keep only amino acid positions
                L = len(seq)
                emb = token_reprs[i, 1 : 1 + L].cpu()  # [L, d_esm]
                esm_embeddings.append(emb)

    return esm_embeddings




def sanitize_sequence(seq: List[str], smiles: List[str]) -> List[str]:
    AA_STD_SET = set("ACDEFGHIKLMNPQRSTVWY")
    new_seq = []
    new_smi = []
    for aa, smile in zip(seq, smiles):
        if '*' in aa:
            new_smi.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=False))
        else:
            new_smi.append(smile)
        temp_aa = aa.replace('*', '')
        if temp_aa.startswith('d') and len(temp_aa) > 1:
            temp_aa = temp_aa[1:]
        if temp_aa in AA_STD_SET:
            new_seq.append(temp_aa)
        else:
            mapped = map_single_smiles_to_aa(smile, topk=1)[0]
            new_seq.append(mapped if (isinstance(mapped, str) and mapped in AA_STD_SET) else 'X')
    return new_seq, new_smi



def getEmbedding(
                    fasta_list: List[List[str]],
                    frag_list: List[List[str]],
                    feature_type: List[str],
                    run_type: str = 'train',
                    activity_array: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
    
 
    rdkit_feats = None
    esm_feats = None
    onehot_feats = None


    clean_fasta_ls = []
    clean_frag_ls = []
    for f, s in zip(fasta_list, frag_list):
        f_tmp, s_tmp = sanitize_sequence(f, s)
        clean_fasta_ls.append(f_tmp)
        clean_frag_ls.append(s_tmp)
 
    save_scaler_path = Path(save_path) / 'rdkit_scaler.pkl'
    if run_type=='train' and save_path:
        scaler = None
    elif run_type=='train' and not save_path:
        print('save_path is required when run_type is train')
        assert 0
    else:
        scaler = joblib.load(save_scaler_path)

    # Feature computation
    if 'RDKit' in feature_type:
        print("Computing RDKit features...")
        # Keep RDKit descriptors on CPU; move to GPU per batch during training
        rdkit_feats = create_rdkit_embedding(clean_frag_ls, save_scaler_path, scaler, device="cpu")

    if 'ESM' in feature_type:
        print("Computing ESM embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        esm_feats = create_esm_embedding(clean_fasta_ls, device=device)
    
    if 'onehot' in feature_type:
        print("Computing onehot features...")
        onehot_feats = create_onehot_embedding(clean_fasta_ls)


    embeddings = []

    for i in range(len(esm_feats)):
        embeddings.append(torch.cat((esm_feats[i], rdkit_feats[i], onehot_feats[i]), dim=1))


    dataset = EmbeddingDataset(
                                embeddings = embeddings,
                                fasta_list = clean_fasta_ls,
                                smiles_list = frag_list,
                                targets = activity_array
                                )

    return dataset




