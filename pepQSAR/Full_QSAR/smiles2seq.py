from rdkit import  Chem
from rdkit.Chem import Draw, rdmolops, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import openbabel
from openbabel import pybel
import re
from itertools import chain
from pathlib import Path

# CHUCKLES representation of amino acids
def convert_to_chuckles(aa_smiles):
    try:
        mol = pybel.readstring('smi', aa_smiles)
        n_term_pat = pybel.Smarts('[$([ND1,ND2]CC(O)=O)]') # Match N-terminal via SMARTS pattern
        c_term_pat = pybel.Smarts('[$([OD1]C(=O)C[ND1,ND2])]') # Match C-terminal via SMARTS pattern
        # 是N端和C端原子的索引
        n_term_idx = n_term_pat.findall(mol)[0][0] 
        c_term_idx = c_term_pat.findall(mol)[0][0]
        # Reorder atoms using OBConversion, set N- and C-terminal atoms as start and end of SMILES
        rearranger = openbabel.OBConversion()
        rearranger.SetInAndOutFormats('smi', 'smi')
        rearranger.AddOption('f', openbabel.OBConversion.OUTOPTIONS, str(n_term_idx)) # Set starting atom of SMILES
        rearranger.AddOption('l', openbabel.OBConversion.OUTOPTIONS, str(c_term_idx)) # Set ending atom of SMILES
        outmol = openbabel.OBMol()
        rearranger.ReadString(outmol, aa_smiles)
        return rearranger.WriteString(outmol).strip()
    except:
        # print('Cannot produce CHUCKLES from', aa_smiles)
        return aa_smiles


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
    if uncharged_aa: 
        uncharged_aa = remove_backbone_charges(uncharged_aa) 
        # Convert to CHUCKLES
        chuckles_aa = convert_to_chuckles(uncharged_aa)
    else:
        chuckles_aa = None
    return chuckles_aa


####### CUT SMILES
def smiles2seq(smiles: str, df: pd.DataFrame):
    """
    
    :param smiles: SMILES of peptide/cyclic peptide
    
    :param df: Table of known amino acid names and SMILES, format: Smiles,Name,Names
    
    :return: List of amino acid names

    """
    uncharger = rdMolStandardize.Uncharger()
    
    def fragmol2aa(frag_mol: Chem.Mol, ss: bool, N_alpha: str, df: pd.DataFrame):
        max_match = 0
        max_subaa = None
        standard_fragsmi = Chem.MolToSmiles(frag_mol, isomericSmiles=False)
        for name, names, smi, inc in df.values:
            # Convert fragments and SMILES in df to non-stereoisomeric form, note: cannot distinguish d-amino acids!
            standard_aamol = Chem.MolFromSmiles(smi)
            Chem.RemoveStereochemistry(standard_aamol)
            standard_aasmi = Chem.MolToSmiles(standard_aamol)
            product = rxn_COO2CO.RunReactants((standard_aamol,))
            standard_aamol = product[0][0] if product else standard_aamol
            matches_aa = frag_mol.GetSubstructMatches(standard_aamol)
            flag_N_alpha = 0
            for match_aa in matches_aa:
                for idx in match_aa:
                    if frag_mol.GetAtomWithIdx(idx).GetProp('atomNote') == N_alpha:
                        flag_N_alpha = 1
                        break
                if flag_N_alpha:
                    break
            if flag_N_alpha:
                if len(match_aa) == frag_mol.GetNumHeavyAtoms() or standard_fragsmi == standard_aasmi:
                    if ss:
                        if names == 'Cys':
                            return 'Cyx'
                        else:
                            raise ValueError(f'Wrong ss index! {Chem.MolToSmiles(frag_mol)}')
                    else:
                        return str(name)
                elif name in ['A', 'G', 'I', 'V', 'S', 'C', 'T']:
                    continue
                elif len(match_aa) >= max_match:
                    max_subaa = str(name)
                    max_match = len(match_aa)
        if max_subaa:
            # print(max_subaa)
            return max_subaa + '*'
        #########
        else:
            # print('xxxxx')
            return 'X'
        ##########
        product = rxn_CO2COO.RunReactants((frag_mol,))
        frag_mol = product[0][0] if product else frag_mol
        return Chem.MolToSmiles(frag_mol)

    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print('Valid input smiles!')
    except:
        
        raise ValueError('Valid input smiles!')
    

    mol = uncharger.uncharge(mol)
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
        
    # Find all atom indices matching two amino acid backbone connections
    smarts_list = []
    
    c_len = 3
    for i in range(c_len):
        for j in range(c_len):
            smarts_list.append('NC' + 'C' * i + 'C(=O)' + 'NC' + 'C' * j + 'C(=O)')
    def N_C_index(match_pair, matches_list):
        for i in range(c_len):
            for j in range(c_len):
                if match_pair in matches_list[i * c_len + j]:
                    return [match_pair[i] for i in [0, 2 + i, 4 + i, -2]]
    # print(smarts_list)
    sub_list = [Chem.MolFromSmarts(smarts) for smarts in smarts_list]
    rxn_CO2COO = AllChem.ReactionFromSmarts('[N,CN:3][*:1][CH1:2](=O)>>[N,CN:3][*:1][*:2](=O)O')
    rxn_COO2CO = AllChem.ReactionFromSmarts('[N,CN:3][*:1][C:2](=O)O>>[N,CN:3][*:1][*:2](=O)')
    matches_list = []
    for sub in sub_list:
        sub_match = mol.GetSubstructMatches(sub)
        if sub_match:
            matches_list.append([list(x) for x in sub_match])
        else:
            matches_list.append([])
    matches = [m for ms in matches_list for m in ms]
    if matches:
        pass
    else:
        return [smiles], [smiles]
    
    heads = []
    head2tail = {}
    # Build linked amino acid chains:
    # head2tail: {current amino acid: [previous amino acid, [next amino acid list]]}, 
    # amino acid: (N-terminal index, C-terminal index)
    for i, match_pair in enumerate(matches):
        flag = False
        nc_indx = N_C_index(match_pair=match_pair, matches_list=matches_list)
        tmp_aa = (nc_indx[0], nc_indx[1])
        next_aa = (nc_indx[2], nc_indx[3])
        tmp_links = head2tail.get(tmp_aa, [None, []])
        tmp_links[1].append(next_aa)
        next_links = head2tail.get(next_aa, [None, []])
        next_links[0] = tmp_aa
        head2tail.update(
            {tmp_aa: tmp_links, next_aa: next_links}
        )
        for j in range(len(matches)):
            # If N-terminal of some amino acid pair appears in other amino acid pairs, it is not a head
            if i != j and tmp_aa[0] in matches[j][3:]:
                flag = True
                break
        if not flag:
            heads.append(tmp_aa)

    # If there is no head, and there is more than one amino acid, process as cyclic peptide with N- and C-terminal amide bonds
    if heads == []:
        nc_indx = N_C_index(matches[0], matches_list)
        heads.append(tuple(nc_indx[:2]))

    # heads may have multiple heads, after cutting, determine which is the main backbone, the rest are not cut as side chains of some amino acid
    def get_sub_seqs(head, top_head, head2tail, mol, df):
        if head is None:
            return [[]], [[]]
        tmp_aa = head
        tmp_sub_seqs = []
        tmp_sub_ss = []
        tmp_sub_smiles = []
        
        tmp_aa_N = mol.GetAtomWithIdx(tmp_aa[0]).GetProp('atomNote')
        last_aa = head2tail[tmp_aa][0]
        next_aa_list = head2tail[tmp_aa][1]
        
        if next_aa_list == []:
            next_aa_list = [None]

        for next_aa in next_aa_list:
            ed_mol = Chem.EditableMol(mol)
            # Cut bond between N-terminal of current amino acid and C(=O) of previous amino acid
            if last_aa:
                ed_mol.RemoveBond(last_aa[1], tmp_aa[0])
            # Cut bond between C(=O) of current amino acid and N-terminal of next amino acid
            if next_aa:
                ed_mol.RemoveBond(tmp_aa[1], next_aa[0])

            broken_mol = ed_mol.GetMol()
            fragments = rdmolops.GetMolFrags(broken_mol, asMols=True, sanitizeFrags=True)
            
            # # If number of fragments after cutting is 2 (non-head/tail) or 1 (head/tail), it contains a side chain ring
            # if len(fragments) == 2 - (tmp_aa == head or next_aa is None):
            for frag in fragments:
                # Currently only consider disulfide bonds
                ss_mode = Chem.MolFromSmarts('[NH2]C(CSSCC(N)C(=O))[CH1](=O)')
                ss_match = frag.GetSubstructMatch(ss_mode)
                ed_mol = Chem.EditableMol(frag)
                if ss_match:
                    ed_mol.RemoveBond(ss_match[3], ss_match[4])
                broken_mol = ed_mol.GetMol()
                sub_fragments = rdmolops.GetMolFrags(broken_mol, asMols=True, sanitizeFrags=True)
                for sub_frag in sub_fragments:
                    flag = 0
                    for atom in sub_frag.GetAtoms():
                        # Find current amino acid in fragment
                        if atom.GetProp('atomNote') == tmp_aa_N:
                            flag = 1
                            break
                    if flag:
                        break
                if flag:
                    break
            
            ss = True if ss_match else False
            tmp_acid = fragmol2aa(sub_frag, ss, tmp_aa_N, df)
            amino_acid = rxn_CO2COO.RunReactants((sub_frag,))
            amino_acid = amino_acid[0][0] if amino_acid else sub_frag
            smiles_frag = Chem.MolToSmiles(amino_acid)
            
            next_seqs, next_ss, next_smiles = ([[]], [[]], [[]]) if next_aa == top_head or next_aa is None \
                else get_sub_seqs(next_aa, top_head, head2tail, mol, df)

            tmp_sub_seqs.extend(
                [[tmp_acid] + sub_seq for sub_seq in next_seqs]
            )
            tmp_sub_ss.extend(
                [[ss] + sub_ss for sub_ss in next_ss]
            )
            tmp_sub_smiles.extend(
                [[smiles_frag] + sub_smi for sub_smi in next_smiles]
            )

        return tmp_sub_seqs, tmp_sub_ss, tmp_sub_smiles
    
    acid_seqs, ss_indexs = [], []
    acid_smiles = []
    for i, head in enumerate(heads):
        next_seqs, next_ss, next_smiles = get_sub_seqs(head, head, head2tail, mol, df)
        acid_seqs.extend(next_seqs)
        ss_indexs.extend(next_ss)
        acid_smiles.extend(next_smiles)

    subseq_lengths = [len(s) for s in acid_seqs]
    max_idx = subseq_lengths.index(max(subseq_lengths))
    ss_indexs = ss_indexs[max_idx]
    acid_seqs = acid_seqs[max_idx]
    acid_smiles = acid_smiles[max_idx]
    acid_chukles = [smiles_to_chuckles(aa) for aa in acid_smiles]    
    return acid_seqs, acid_chukles





def DL_change(name_to_smiles, inchi_to_name, aa_seqs_ls, aa_smiles_ls):

    # Precompute InChI and SMILES patterns of amino acids
    aa_data = []
    for smi in aa_smiles_ls:
        mol = Chem.MolFromSmiles(smi)
        inchi = Chem.MolToInchi(mol)
        aa_data.append({'mol': mol, 'inchi': inchi})
    
    aa_seqs_ls_new = []
    aa_smis_ls_new = []
    for seq, data, smi in zip(aa_seqs_ls, aa_data, aa_smiles_ls):
        if seq == 'X':
            new_seq = seq # Not in our custom amino acid list

        else:
            new_seq = inchi_to_name.get(data['inchi'], 0)
            if new_seq: # Fully matches our custom amino acid list
                pass 
            elif '*' in seq: # Fuzzy matches our custom amino acid list
                # Check chirality
                seq_type = seq[:-1]  # Remove *
                new_seq = process_chiral_sequence(seq, seq_type, data['mol'], name_to_smiles)
            else:
                seq_type = seq # Differences in chirality with our custom amino acids
                new_seq = process_chiral_sequence(seq, seq_type, data['mol'], name_to_smiles)
        new_smi = name_to_smiles.get(new_seq, smi) 
        aa_seqs_ls_new.append(new_seq)
        aa_smis_ls_new.append(new_smi)     
    return aa_seqs_ls_new, aa_smis_ls_new


def process_chiral_sequence(seq, seq_type, mol, name_to_smiles):
    """processing chiral sequences"""  
    pattern_smiles = name_to_smiles[seq_type]
    aa_mol = Chem.MolFromSmiles(pattern_smiles) # Amino acid structure
    aa_gene_pattern = Chem.MolFromSmarts('[NH2]-[C@H,C@@H](-C(=O)O)') # Amino acid structure general formula
    
    # Check chiral centers
    aa_chiral_centers = Chem.FindMolChiralCenters(aa_mol, force=True)
    aa_gene_matches = set(chain(*(aa_mol.GetSubstructMatches(aa_gene_pattern))))
    
    if len(aa_chiral_centers)==0:
        return seq[1:] if seq[0] == 'd' else seq
    else:
        for atom_id, chiral in aa_chiral_centers:
            if atom_id in aa_gene_matches:
                aa_pattern_chiral = chiral
        
    
    # Find matching patterns
    mol_matches = set(chain(*(mol.GetSubstructMatches(aa_mol))))
    mol_gene_matches = set(chain(*(mol.GetSubstructMatches(aa_gene_pattern))))
    matches = mol_matches & mol_gene_matches
    mol_chiral_centers = Chem.FindMolChiralCenters(mol, force=True)

    if len(matches)==0:
        return seq[1:] if seq[0] == 'd' else seq
  
    for atom_id, chiral in mol_chiral_centers:
        if atom_id in matches and chiral != aa_pattern_chiral:
            return seq[1:] if seq[0] == 'd' else 'd' + seq
    
    return seq[1:] if seq[0] == 'd' else seq



def peptide_smiles2seq(smiles):
    script_path = Path(__file__).resolve()
    aa_all = pd.read_csv(script_path.parent / 'cut_AA_183.txt', usecols=['Chuckles','Name','Names', 'InChi'])
    name_to_smiles = dict(zip(aa_all['Name'], aa_all['Chuckles']))
    inchi_to_name = dict(zip(aa_all['InChi'], aa_all['Name']))

    # Cut SMILES into amino acids, cannot distinguish chiral amino acids
    acid_seqs, acid_chukles = smiles2seq(smiles, aa_all)

    # Correct amino acid chirality using SMILES after cutting
    acid_seqs_update, acid_smiles_updata = DL_change(name_to_smiles, inchi_to_name, acid_seqs, acid_chukles)

    # Map back to original sequence after terminal modifications
    func_seq_ls = acid_seqs_update.copy()
    func_seq_smiles = acid_smiles_updata.copy()

    # Map back to modified amino acid representation for C- and N-terminal modifications
    if 'ac' in acid_seqs_update[0]:
        func_seq_ls[0] = acid_seqs_update[0].replace('ac', '')
        func_seq_smiles[0] = name_to_smiles.get(func_seq_ls[0], func_seq_smiles[0])
    if 'am' in acid_seqs_update[-1]:
        func_seq_ls[-1] = acid_seqs_update[-1].replace('am', '')
        func_seq_smiles[-1] = name_to_smiles.get(func_seq_ls[-1], func_seq_smiles[-1])
    
    return func_seq_ls, func_seq_smiles, acid_seqs_update, acid_smiles_updata

