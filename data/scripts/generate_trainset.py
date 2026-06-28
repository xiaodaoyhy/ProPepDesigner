
from rdkit import RDLogger
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import openbabel
from openbabel import pybel
from typing import List
import re
RDLogger.DisableLog('rdApp.*')



# CHUCKLES representation of amino acids
def convert_to_chuckles(aa_smiles):
    try:
        mol = pybel.readstring('smi', aa_smiles)
        n_term_pat = pybel.Smarts('[$([ND1,ND2]CC(O)=O)]')  # Match N-terminus via SMARTS
        c_term_pat = pybel.Smarts('[$([OD1]C(=O)C[ND1,ND2])]')  # Match C-terminus via SMARTS
        # Indices of N- and C-terminal atoms
        n_term_idx = n_term_pat.findall(mol)[0][0] 
        c_term_idx = c_term_pat.findall(mol)[0][0]
        # Reorder atoms with OBConversion; set N- and C-termini as SMILES start/end
        rearranger = openbabel.OBConversion()
        rearranger.SetInAndOutFormats('smi', 'smi')
        rearranger.AddOption('f', openbabel.OBConversion.OUTOPTIONS, str(n_term_idx))  # SMILES start atom
        rearranger.AddOption('l', openbabel.OBConversion.OUTOPTIONS, str(c_term_idx))  # SMILES end atom
        outmol = openbabel.OBMol()
        rearranger.ReadString(outmol, aa_smiles)
        return rearranger.WriteString(outmol).strip()
    except:
        print('Cannot produce CHUCKLES from', aa_smiles)
        assert 0


def uncharger(smiles):
    mol = Chem.MolFromSmiles(smiles)
    uncharger = rdMolStandardize.Uncharger() 
    if mol:
        uncharged_mol = uncharger.uncharge(mol)
        uncharged_smi = Chem.MolToSmiles(uncharged_mol, isomericSmiles=True)
    else:
        print('error:', smiles)
        uncharged_smi = None
    return uncharged_smi

def remove_backbone_charges(original_smiles):
    if '[O-]' in original_smiles:
        modified_smiles = re.sub(r'\[O\-\]', 'O', original_smiles)
    else:
        modified_smiles = original_smiles
    return modified_smiles

def smiles_to_chuckles(amino_acid):
    # Preprocessing
    uncharged_aa = uncharger(amino_acid)  # Remove charges
    uncharged_aa = remove_backbone_charges(uncharged_aa)  # Remove backbone charges
    # Convert to CHUCKLES
    chuckles_aa = convert_to_chuckles(uncharged_aa)
    return chuckles_aa

def peptide2chuckles(amino_acids):
    aas = [smiles_to_chuckles(aa) for aa in amino_acids]
    return aas




# Build linear peptide topology
def linear(smiles):
    smiles[-1] = smiles[-1] +'O'
    return smiles
    
def assign_topology(peptide_chuckles, topology, cyclization_info=None):
    '''Converts list of amino acid smiles into amino acids chuckles with the desired topological information
    
        peptide_chuckles: 
                A list of amino acids ordered according to their position in the intended peptide chain
        
        topology: 
                The choice of topology from the selection 
                [ Linear, Head-To-Tail, Sidechain-To-Tail, Head-To-Sidechain, Disulfide-Bridge ]
        
        cyclization_info: 
                A list of the positions of the chosen amino acids participating in the peptide cyclization. 
                Is only necessary for Sidechain-To-Tail, Head-To-Sidechain, Disulfide-Bridge
                '''
    if topology == 'Linear':
        peptide_with_topology = linear(peptide_chuckles)
    else:
        print(f'Topology {topology} is not yet supported')
    return peptide_with_topology


def mask_peptide(peptide_chuckles, positions):
    peptide = peptide_chuckles.copy()
    masked_aas = ["?" if idx in positions else x for idx, x in enumerate(peptide)]
    return masked_aas

def merge_peptide(peptide_chuckles):
    peptide = [x[:-1] if x!="?" else x for x in peptide_chuckles]
    peptide = "|".join(peptide)
    return peptide

def merge_original_peptide(peptide_chuckles):
    peptide = [x[:-1] for x in peptide_chuckles]
    peptide = "".join(peptide)
    return peptide

def build_peptide(amino_acids: List, topology: str, masking_positions: List, cyclization_info=None):
    ''' Builds the peptide from a list of amino acids'''
    aa_chuckles = peptide2chuckles(amino_acids) 
    peptide_chuckles = assign_topology(aa_chuckles, topology, cyclization_info)
    masked_chuckles = mask_peptide(peptide_chuckles, masking_positions)
    
    input_peptide = merge_peptide(masked_chuckles)
    original_peptide = merge_original_peptide(peptide_chuckles)
    
    return original_peptide, input_peptide





## Training set generation
from re import T
import numpy as np
from rdkit import Chem
import random
from math import ceil

def sample_truncated_normal(mean: float, stddev: float, low: float, high: float) -> float:
    """Sample a value from N(mean, stddev^2) truncated to [low, high]."""
    for _ in range(100):
        value = np.random.normal(mean, stddev)
        if low <= value <= high:
            return float(value)
    return float(np.clip(np.random.normal(mean, stddev), low, high))


def generate_peptide(natural_aas, nnAA_1w_dict, nnAA_target_dict, all_aa_library, seed):
    np.random.seed(seed)
    # 1. Select topology
    topology = np.random.choice(["Linear", "head-to-tail", "disulfide", "sidechain-tail"], 
                               p=[1, 0, 0, 0])
    
    # 2. Generate peptide length

    length = np.random.randint(10, 45)
    

    # 3. Choose NNAA fraction via truncated normal sampling
    # p_nnaa = sample_truncated_normal(mean=0.15, stddev=0.08, low=0.0, high=0.3)
    p_nnaa = np.random.uniform(0, 0.35)
    nnaa_num = int(np.round(length * p_nnaa))
    



    # 4. Sample amino acids (favor nnAA_target over nnAA_1w within NNAAs)
    natural_count = length - nnaa_num
    natural_names = np.random.choice(list(natural_aas.keys()), natural_count, replace=True)

    if nnaa_num > 0:
        target_count = np.random.binomial(n=nnaa_num, p=0.8)  # Favor nnAA_target over nnAA_1w within NNAAs
    else:
        target_count = 0
    onew_count = max(0, nnaa_num - target_count)

    target_names = np.random.choice(list(nnAA_target_dict.keys()), target_count, replace=True) if target_count > 0 else []
    onew_names = np.random.choice(list(nnAA_1w_dict.keys()), onew_count, replace=True) if onew_count > 0 else []

    aas_seq = list(natural_names) + list(target_names) + list(onew_names)
    np.random.shuffle(aas_seq)  # Shuffle order
    aas = [all_aa_library[k] for k in aas_seq]
    

    # 5. Add modifications: sample modification event probability and fraction from distributions
    new_aa_library = {}
    # event_prob = np.random.beta(MOD_EVENT_BETA_A, MOD_EVENT_BETA_B)
    event_prob = 0.1
    if np.random.rand() < event_prob:
        # mod_ratio = np.random.beta(MOD_RATIO_BETA_A, MOD_RATIO_BETA_B)
        mod_ratio = np.clip(np.random.normal(0.05, 0.02), 0, 0.1)
        # modification_count = max(1, np.random.binomial(n=len(aas), p=mod_ratio))
        modification_count = max(1, int(round(len(aas)*mod_ratio)))
        if modification_count > 0:
            modification_positions = np.random.choice(len(aas), modification_count, replace=False)
            for pos in modification_positions:
                aa = aas[pos]
                aa_name = aas_seq[pos]
                # Randomly choose modification type: methylation or stereoinversion (methylation preferred)
                if np.random.rand() < 0.8 and (aa[:2] != 'N1'):  # Methylation (exclude Pro)
                    aas[pos] = smiles_to_chuckles("N(C)" + aa[1:])
                    if aas[pos] not in all_aa_library.values():
                        new_aa_library[f'{aa_name}Me'] = aas[pos]
                        aas_seq[pos] = f'{aa_name}Me'
                elif '@' in aa:  # Stereoinversion (if amino acid has a chiral center)
                    if aa[:6] == 'N[C@H]':
                        aas[pos] = smiles_to_chuckles('N[C@@H]' + aa[6:])
                    elif aa[:7] == 'N[C@@H]':
                        aas[pos] = smiles_to_chuckles('N[C@H]' + aa[7:])
                    else:
                        continue
                    if aas[pos] not in all_aa_library.values():
                        new_aa_library[f'd{aa_name}'] = aas[pos]
                        aas_seq[pos] = f'd{aa_name}'
    
    # 6. Randomly choose mask positions; mask count per peptide in [10%, 90%] of length
    natural_aa_indices = [i for i,aa in enumerate(aas) if aa in natural_aas.values()]  # Track natural AA positions
    masked_positions = mask_positions(peptide_length=len(aas), natural_aa_indices=natural_aa_indices)

    # 7. Build peptide_seq and peptide_chuckles
    query_peptide, source_peptide = build_peptide(aas, topology, masked_positions)
    target_peptide = [aas[j][:-1] if j!=len(aas)-1 else aas[j] for j in masked_positions]
    target_peptide = "|".join(target_peptide)

    data = {'Query_Mol':[query_peptide],'Source_Mol':[source_peptide],\
            'Target_Mol':[target_peptide], 'length': [length], 'sequence': ["-".join(aas_seq)]}
    return data, new_aa_library



def mask_positions(peptide_length, natural_aa_indices=None):
    max_mask_ratio = np.random.uniform(0.1, 0.4)
    # mask_num = max(int(np.round(peptide_length * 0.2)), int(np.round(peptide_length * max_mask_ratio)))
    mask_num = int(np.round(peptide_length * max_mask_ratio))
    
    all_positions = np.arange(peptide_length)
    natural_mask_num = int(mask_num * (np.random.beta(a=2, b=5, size=1) * 0.5).item())  # Fraction of natural AA masks
    nnaa_mask_num = mask_num - natural_mask_num
    # Split natural vs. non-natural amino acids
    if natural_aa_indices:
        nn_aa_indices = list(set(all_positions) - set(natural_aa_indices))
        if len(nn_aa_indices):
            replace_aa = len(natural_aa_indices) < natural_mask_num
            masked_natural = np.random.choice(natural_aa_indices, natural_mask_num, replace=replace_aa)
            replace_nn_aa = len(nn_aa_indices) < nnaa_mask_num
            masked_nnaa = np.random.choice(nn_aa_indices, nnaa_mask_num, replace=replace_nn_aa)
            masked_positions = np.concatenate([masked_natural, masked_nnaa]).astype(int)
        else:
            masked_positions = np.random.choice(natural_aa_indices, mask_num, replace=False)           
        # print('masked_natural:',masked_natural, 'masked_nnaa:',masked_nnaa,'masked_positions:', masked_positions, 'peptide_length:', peptide_length, 'natural AA count:', len(natural_aa_indices))
    else:
        masked_positions = np.random.choice(all_positions, mask_num, replace=False)

    return sorted(masked_positions)



from concurrent.futures import ProcessPoolExecutor
import json
from collections import defaultdict
import glob
import os
from pathlib import Path

def process_seed(s):
    np.random.seed(s)
    random.seed(s)
    return generate_peptide(natural_aas, nnAA_1w_dict, nnAA_target_dict, all_aa_library, s)


# Load amino acid library
nnAA_1w = pd.read_csv('10000NNAAs.txt', sep=' ')  # 9998 entries
nnAA_target = pd.read_csv('cut_AA_38.txt', sep=',')  # 58 entries
nAA = pd.read_csv('20AAs.txt', sep=',')


nnAA_1w['Chuckles'] = peptide2chuckles(nnAA_1w['Smiles'])
nnAA_target['Chuckles'] = peptide2chuckles(nnAA_target['Smiles'])
nAA['Chuckles'] = peptide2chuckles(nAA['Smiles'])
print('CHUCKLES conversion succeeded')
all_nnaa_df = pd.concat([nnAA_1w, nnAA_target], axis=0)
all_aa_df = pd.concat([nnAA_1w, nnAA_target, nAA], axis=0)


# Natural / NNAA libraries (extend as needed)
natural_aas = {k: v for k, v in zip(nAA['Name'], nAA['Chuckles'])}
nnAA_1w_dict = {k: v for k, v in zip(nnAA_1w['Name'], nnAA_1w['Chuckles'])}
nnAA_target_dict = {k: v for k, v in zip(nnAA_target['Name'], nnAA_target['Chuckles'])}
all_aa_library = {k: v for k, v in zip(all_aa_df['Name'], all_aa_df['Chuckles'])}



# 1. Preallocate data structures
new_all_aa_dict = all_aa_library
chunk_size = 10000
output_dir = "temp"
os.makedirs(output_dir, exist_ok=True)

# 2. Chunk processing function
def process_and_save(chunk):
    print('chunk')
    temp_dict = defaultdict(list)
    for data, new_aa in chunk:
        for k, v in data.items():
            temp_dict[k].extend(v)
        new_all_aa_dict.update(new_aa)
    # print(temp_dict)
    pd.DataFrame(temp_dict).to_csv(f"{output_dir}/chunk_{id(chunk)}.csv")

# 3. Parallel processing (workers return results; main process merges and updates new_all_aa_dict)
with ProcessPoolExecutor(max_workers=80) as executor:
    futures = []
    for s in range(520000):
        futures.append(executor.submit(process_seed, s))
        if len(futures) >= chunk_size:
            chunk = [f.result() for f in futures]
            process_and_save(chunk)  # Write and update synchronously
            futures = []

    # Process remaining tasks
    if futures:
        process_and_save([f.result() for f in futures])

# 4. Merge results
all_data = pd.concat(
    [pd.read_csv(f) for f in glob.glob(f"{output_dir}/chunk_*.csv")],
    ignore_index=True
)
output_file = Path(output_dir).parent / "trainset_merged_data.csv"
all_data.to_csv(output_file, index=False)

# Write output files
with open(Path(output_dir).parent / 'base_trainset_new_all_aa_dict.json', 'w') as f:
    json.dump(new_all_aa_dict, f, indent=4) 