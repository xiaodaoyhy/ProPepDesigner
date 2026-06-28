
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import openbabel
from openbabel import pybel
from typing import List
import re
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split
sys.path.append('../../GenePeptide')
from reinvent_models.mol2mol.models.vocabulary import SMILESTokenizer


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
    
def assign_topology(peptide_chuckles, topology):
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

def build_peptide(amino_acids: List, topology: str, masking_positions: List):
    ''' Builds the peptide from a list of amino acids'''
    aa_chuckles = peptide2chuckles(amino_acids) 
    peptide_chuckles = assign_topology(aa_chuckles, topology)
    masked_chuckles = mask_peptide(peptide_chuckles, masking_positions)
    
    input_peptide = merge_peptide(masked_chuckles)
    original_peptide = merge_original_peptide(peptide_chuckles)
    
    return original_peptide, input_peptide



def generate_peptide(aa_smiles_ls):
    # 1. Select topology
    topology = np.random.choice(["Linear", "head-to-tail", "disulfide", "sidechain-tail"], 
                               p=[1, 0, 0, 0])
    
    # 2. Randomly choose mask positions; mask count per peptide in [10%, 90%] of length
    masked_positions = mask_positions(peptide_length=len(aa_smiles_ls), natural_aa_indices=None)

    # 3. Build peptide_seq and peptide_chuckles
    query_peptide, source_peptide = build_peptide(aa_smiles_ls, topology, masked_positions)
    target_peptide = [aa_smiles_ls[j][:-1] if j!=len(aa_smiles_ls)-1 else aa_smiles_ls[j] for j in masked_positions]
    target_peptide = "|".join(target_peptide)

    data = {'Query_Mol':[query_peptide],'Source_Mol':[source_peptide],\
            'Target_Mol':[target_peptide], 'length': [len(aa_smiles_ls)]}

    return data



def mask_positions(peptide_length, natural_aa_indices=None):
    max_mask_ratio = np.random.uniform(0.1, 0.5)
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

    else:
        masked_positions = np.random.choice(all_positions, mask_num, replace=False)

    return sorted(masked_positions)



if __name__ == "__main__":
    # Load amino acid library
    nnAA_1w = pd.read_csv('../result/10000NNAAs.txt', sep=',')  # 9998 entries
    nnAA_target_raw = pd.read_csv('../../myutils/cut_AA_183.txt', sep=',')
    nnAA_target = nnAA_target_raw[~nnAA_target_raw['Name'].str.contains('am', na=False)].reset_index(drop=True)
    nnAA_1w['Chuckles'] = peptide2chuckles(nnAA_1w['Smiles'])
    nnAA_target['Chuckles'] = peptide2chuckles(nnAA_target['Smiles'])
    all_aa_df = pd.concat([nnAA_1w, nnAA_target], axis=0)
    all_aa_library = {k: v for k, v in zip(all_aa_df['Name'], all_aa_df['Chuckles'])}
    print(f"Amino acid library size: {all_aa_df.shape}")
    print('CHUCKLES conversion succeeded')

    save_dir = Path('../result/')
    save_dir.mkdir(parents=True, exist_ok=True)

    df_target_init = pd.read_csv('../result/TargetData_removeLongside.csv', usecols=['pep_sequence_model', 'length'])
    df_target = df_target_init[df_target_init['length'].between(10, 45)]  # Keep peptides of length 10-45

    # Drop rows with empty pep_sequence
    df_target = df_target[df_target['pep_sequence_model'].notna()].reset_index(drop=True)


    # Convert to backbone amino acid sequence
    df_target['main_sequence'] = df_target['pep_sequence_model'].apply(lambda x: x.replace('*',''))


    # Get unique backbone amino acid sequences
    df_target_unique = df_target.drop_duplicates(subset=['main_sequence'], keep='first', inplace=False)
    valid_peptides = df_target_unique[df_target_unique['main_sequence'].notna()].reset_index(drop=True)
    print(f'Unique backbone sequences: {len(df_target_unique)}')
    print(len(valid_peptides))


    # Convert backbone sequences to CHUCKLES
    valid_peptides['main_sequence_chuckles'] = valid_peptides['main_sequence'].apply(lambda x: [all_aa_library[aa] for aa in eval(x)])


    # Build PepINVENT input sequences; repeat several times for more data
    temp_dict = defaultdict(list)
    s = 0
    for n in range(5):
        for i in range(len(valid_peptides)):
            data = generate_peptide(valid_peptides.loc[i,'main_sequence_chuckles'])
            s = s+1
            temp_dict['main_sequence'].append(valid_peptides.loc[i,'main_sequence'])
            for k, v in data.items():
                temp_dict[k].extend(v)


    df_concat = pd.DataFrame(temp_dict)  # Converted sequences
    df = df_concat.drop_duplicates(subset=['Source_Mol'], keep='first', inplace=False).reset_index(drop=True)
    print(f"Generated {len(df_concat)} masked peptides; {len(df)} after deduplication")


    tokenizer = SMILESTokenizer()
    max_token_length = [df['Target_Mol'].apply(lambda x: len(tokenizer.tokenize(x))).max(), df['Source_Mol'].apply(lambda x: len(tokenizer.tokenize(x))).max()]
    print(max_token_length)

    strata = pd.cut(df['length'], bins=np.arange(5, 55, 5))
    train_df, tmp_df = train_test_split(df, test_size=0.2, stratify=strata, random_state=42)
    tmp_strata = pd.qcut(tmp_df['length'], q=strata.nunique(), labels=False, duplicates='drop')
    val_df, test_df = train_test_split(tmp_df, test_size=0.2, stratify=tmp_strata, random_state=42)

    # Split into train, validation, and test sets
    print(f"trainset={len(train_df)}, validation={len(val_df)}, testset={len(test_df)}")
    # Save as CSV files
    train_df.to_csv(save_dir / 'finetune_train.csv', index=False)
    val_df.to_csv(save_dir / 'finetune_valid.csv', index=False)
    test_df.to_csv(save_dir / 'finetune_test.csv', index=False)



