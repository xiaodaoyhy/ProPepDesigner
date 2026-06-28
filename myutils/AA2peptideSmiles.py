from rdkit import RDLogger
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import openbabel
from openbabel import pybel
from typing import List
import re
RDLogger.DisableLog('rdApp.*')



# 氨基酸的CHUCKLES表示
def convert_to_chuckles(aa_smiles):
    try:
        mol = pybel.readstring('smi', aa_smiles)
        n_term_pat = pybel.Smarts('[$([ND1,ND2]CC(O)=O)]') # 通过SMARTS模式匹配 N 端
        c_term_pat = pybel.Smarts('[$([OD1]C(=O)C[ND1,ND2])]') # 通过SMARTS模式匹配 C 端
        # 是N端和C端原子的索引
        n_term_idx = n_term_pat.findall(mol)[0][0]
        c_term_idx = c_term_pat.findall(mol)[0][0]
        # 使用OBConversion重排原子顺序，将N端和C端原子设为SMILES的起点和终点
        rearranger = openbabel.OBConversion()
        rearranger.SetInAndOutFormats('smi', 'smi')
        rearranger.AddOption('f', openbabel.OBConversion.OUTOPTIONS, str(n_term_idx)) # 指定SMILES的起始原子
        rearranger.AddOption('l', openbabel.OBConversion.OUTOPTIONS, str(c_term_idx)) # 指定SMILES的结束原子
        outmol = openbabel.OBMol()
        rearranger.ReadString(outmol, aa_smiles)
        return rearranger.WriteString(outmol).strip()
    except:
        # print('Cannot produce CHUCKLES from', aa_smiles)
        return None


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
    uncharged_aa = uncharger(amino_acid)
    if uncharged_aa:  # 处理带电荷的结构
        uncharged_aa = remove_backbone_charges(uncharged_aa) # 处理带电荷的结构
        # Convert to CHUCKLES
        chuckles_aa = convert_to_chuckles(uncharged_aa)
    else:
        chuckles_aa = None
    return chuckles_aa

def peptide2chuckles(amino_acids):
    aas = [smiles_to_chuckles(aa) for aa in amino_acids]
    return aas


# peptide 线性拓扑的构建
def linear(smiles):
    smiles[-1] = smiles[-1] +'O'
    return smiles


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

def build_peptide(amino_acids: List,  masking_positions: List):
    ''' Builds the peptide from a list of amino acids'''
    aa_chuckles = peptide2chuckles(amino_acids)
    peptide_chuckles = linear(aa_chuckles)
    masked_chuckles = mask_peptide(peptide_chuckles, masking_positions)

    input_peptide = merge_peptide(masked_chuckles)
    original_peptide = merge_original_peptide(peptide_chuckles)

    return original_peptide, input_peptide