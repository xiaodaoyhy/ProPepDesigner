from rdkit import  Chem
from rdkit.Chem import Draw, rdmolops, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from rdkit import RDLogger
# 禁用所有rdApp.*相关的警告
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import openbabel
from openbabel import pybel
import re
from itertools import chain
from pathlib import Path

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
    if uncharged_aa:  # 处理带电荷的结构
        uncharged_aa = remove_backbone_charges(uncharged_aa) # 处理带电荷的结构
        # Convert to CHUCKLES
        chuckles_aa = convert_to_chuckles(uncharged_aa)
    else:
        chuckles_aa = None
    return chuckles_aa


####### 切割分子
def smiles2seq(smiles: str, df: pd.DataFrame):
    """
    
    :param smiles: 多肽/环肽的smiles
    
    :param df: 已知氨基酸名称和smiles对应表格, 格式: Smiles,Name,Names
    
    :return: 氨基酸全称列表

    """
    uncharger = rdMolStandardize.Uncharger()
    
    def fragmol2aa(frag_mol: Chem.Mol, ss: bool, N_alpha: str, df: pd.DataFrame):
        max_match = 0
        max_subaa = None
        standard_fragsmi = Chem.MolToSmiles(frag_mol, isomericSmiles=False)
        for name, names, smi, inc in df.values:
            # 把片段和df中的smiles都转成无立体异构的形式, 注: 因此无法区分d型氨基酸!
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
    except Exception:
        print('Valid input smiles!')
        return None, None
    if mol is None:
        print('Invalid input smiles!')
        return None, None
    

    mol = uncharger.uncharge(mol)
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
        
    # 找到所有匹配两个氨基酸骨架相连的原子index
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
    # 构建前,中,后相连接的氨基酸链, head2tail: {当前氨基酸: [上一个氨基酸, [下一个氨基酸列表]]}, 氨基酸: (N端index, C端index)
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
            # 如果某个氨基酸对中的N端出现在其他的氨基酸对中，则认为不是头部
            if i != j and tmp_aa[0] in matches[j][3:]:
                flag = True
                break
        if not flag:
            heads.append(tmp_aa)

    # 如果没有头部, 而且不止一个氨基酸, 就按照(首尾酰胺键单环)环肽处理
    if heads == []:
        nc_indx = N_C_index(matches[0], matches_list)
        heads.append(tuple(nc_indx[:2]))

    # heads中可能有多个头部, 进行切割后判断哪个更长就视为主骨架, 其余的不切割作为某个氨基酸的侧链
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
            # 切割当前氨基酸的N和上一个氨基酸的C(=O)之间的连键
            if last_aa:
                ed_mol.RemoveBond(last_aa[1], tmp_aa[0])
            # 切割当前氨基酸的C(=O)和下一个氨基酸的N之间的连键
            if next_aa:
                ed_mol.RemoveBond(tmp_aa[1], next_aa[0])

            broken_mol = ed_mol.GetMol()
            fragments = rdmolops.GetMolFrags(broken_mol, asMols=True, sanitizeFrags=True)
            
            # # 如果切割完片段数在非头尾时是2或者在头尾出是1, 则认为包含侧链成环
            # if len(fragments) == 2 - (tmp_aa == head or next_aa is None):
            for frag in fragments:
                # 目前只考虑二硫键
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
                        # 在片段中找出当前氨基酸
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

    # 预计算氨基酸的InChI和SMILES模式
    aa_data = []
    for smi in aa_smiles_ls:
        mol = Chem.MolFromSmiles(smi)
        inchi = Chem.MolToInchi(mol)
        aa_data.append({'mol': mol, 'inchi': inchi})
    
    aa_seqs_ls_new = []
    aa_smis_ls_new = []
    for seq, data, smi in zip(aa_seqs_ls, aa_data, aa_smiles_ls):
        if seq == 'X':
            new_seq = seq # 不在我们自定义的氨基酸列表中

        else:
            new_seq = inchi_to_name.get(data['inchi'], 0)
            if new_seq: # 完全匹配我们自定义的氨基酸列表中
                pass 
            elif '*' in seq: # 模糊匹配我们自定义的氨基酸列表中
                # 检查手性
                seq_type = seq[:-1]  # 去掉*
                new_seq = process_chiral_sequence(seq, seq_type, data['mol'], name_to_smiles)
            else:
                seq_type = seq # 与我们自定义的氨基酸手性存在差别
                new_seq = process_chiral_sequence(seq, seq_type, data['mol'], name_to_smiles)
        new_smi = name_to_smiles.get(new_seq, smi) 
        aa_seqs_ls_new.append(new_seq)
        aa_smis_ls_new.append(new_smi)     
    return aa_seqs_ls_new, aa_smis_ls_new


def process_chiral_sequence(seq, seq_type, mol, name_to_smiles):
    """处理手性序列的逻辑"""
    pattern_smiles = name_to_smiles[seq_type]
    aa_mol = Chem.MolFromSmiles(pattern_smiles) # 氨基酸结构
    aa_gene_pattern = Chem.MolFromSmarts('[NH2]-[C@H,C@@H](-C(=O)O)') # 氨基酸结构通式部分

    # 检查手性中心
    aa_chiral_centers = Chem.FindMolChiralCenters(aa_mol, force=True)
    aa_gene_matches = set(chain(*(aa_mol.GetSubstructMatches(aa_gene_pattern))))

    if len(aa_chiral_centers)==0:
        return seq[1:] if seq[0] == 'd' else seq

    for atom_id, aa_chiral in aa_chiral_centers:
        if atom_id in aa_gene_matches:
            # 查找匹配的模式
            mol_matches = set(chain(*(mol.GetSubstructMatches(aa_mol))))
            mol_gene_matches = set(chain(*(mol.GetSubstructMatches(aa_gene_pattern))))
            matches = mol_matches & mol_gene_matches
            mol_chiral_centers = Chem.FindMolChiralCenters(mol, force=True)

            if len(matches)==0:
                return seq[1:] if seq[0] == 'd' else seq

            for atom_id, chiral in mol_chiral_centers:
                if atom_id in matches and chiral != aa_chiral:
                    return seq[1:] if seq[0] == 'd' else 'd' + seq

    return seq[1:] if seq[0] == 'd' else seq



def peptide_smiles2seq(smiles):
    script_path = Path(__file__).resolve()
    aa_all = pd.read_csv(script_path.parent / 'cut_AA_183.txt', usecols=['Smiles','Name','Names', 'InChi'])
    name_to_smiles = dict(zip(aa_all['Name'], aa_all['Smiles']))
    inchi_to_name = dict(zip(aa_all['InChi'], aa_all['Name']))

    # 将smiles切割为氨基酸，此时无法区分氨基酸残基的手性
    acid_seqs, acid_chukles = smiles2seq(smiles, aa_all)
    if acid_seqs is None or acid_chukles is None:
        return None

    # 通过切割后的SMILES来修正氨基酸的手性
    acid_seqs_update, acid_smiles_updata = DL_change(name_to_smiles, inchi_to_name, acid_seqs, acid_chukles)

    # 处理末端修饰 映射回原始的序列
    func_seq_ls = acid_seqs_update.copy()
    func_seq_smiles = acid_smiles_updata.copy()

    # 将C端修饰和N端修饰映射回修饰的氨基酸表示
    if 'ac' in acid_seqs_update[0]:
        func_seq_ls[0] = acid_seqs_update[0].replace('ac', '')
        func_seq_smiles[0] = name_to_smiles.get(func_seq_ls[0], func_seq_smiles[0])
    if 'am' in acid_seqs_update[-1]:
        func_seq_ls[-1] = acid_seqs_update[-1].replace('am', '')
        func_seq_smiles[-1] = name_to_smiles.get(func_seq_ls[-1], func_seq_smiles[-1])
    
    return func_seq_ls, func_seq_smiles, acid_seqs_update, acid_smiles_updata

