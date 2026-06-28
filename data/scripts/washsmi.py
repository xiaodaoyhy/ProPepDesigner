
# -*- coding: UTF-8 -*-

# 创建：殷红岩 20250315  rdikt==2024.09   
# 分子结构清洗和标准化工具

# 功能：
# - 验证SMILES的可识别性
# - 处理混合物和盐离子
# - 中和电荷
# - 生成标准SMILES
# - 识别分子重复

# 该脚本一般运行二次，
# 输入：xxx.csv                                               
# 输出：增加了washsmi和check列的xxx.csv文件

# 使用方法：
# 1. 第一次运行（不保留立体化学），为了能够检查到有立体化学的分子对活性的影响:
#    python washsmi_v3.py -i test_washsmi.csv -o test_washsmi_temp1.csv -iso

# 2. 第二次运行（保留立体化学），:
#    python washsmi_v3.py -i test_washsmi.csv
# 


import pandas as pd
import argparse
from rdkit.Chem import PandasTools as pt
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



def washsmi(smi, iso):
    try:
        clean_mol = Chem.MolFromSmiles(smi)
        clean_mol = rdMolStandardize.Cleanup(clean_mol) #除去氢、对非标准价键进行标准化处理
        #仅保留主要片段作为分子,但有时效果不理想，需要人工检查
        clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        #中性化处理分子
        clean_mol = rdMolStandardize.ChargeParent(clean_mol)
        new_smi = Chem.MolToSmiles(clean_mol, isomericSmiles=iso)
    except:
        new_smi = None
        print('Invalid smiles: {}'.format(smi))
    return new_smi


def washToInChi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        InChi = Chem.MolToInchi(mol)
    except:
        InChi = None
        print('Invalid smiles: {}'.format(smi))
    return InChi


# def InChiTosmi(InChi, isomericSmiles=True):
#     if InChi != None:
#         mol = Chem.MolFromInchi(InChi)
#         stand_smi = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
#     else:
#         stand_smi = None
#     return stand_smi



def process_molecules(df, smi_column, iso):
    if iso:
        washsmi_col = 'washsmi_Iso'
    else:
        washsmi_col = 'washsmi_DelIso'

    df[f'{washsmi_col}'] = df[smi_column].apply(lambda x: washsmi(x, iso))
    df['InChi'] = df[f'{washsmi_col}'].apply(lambda x: washToInChi(x))
    
    # df[f'{washsmi_col}'] = df['InChi'].apply(lambda x: InChiTosmi(x, iso))
    
    # 标记无效分子
    df['check'] = None
    nan_idx = df[df[f'{washsmi_col}'].isna()].index
    df.loc[nan_idx, 'check'] = -1
    # 标记重复分子
    df_dup = df[df.duplicated(subset='InChi', keep='first')] #将重复项标记为除第一次出现的分子
    df_dup_na = df_dup.dropna(subset=['InChi'], inplace=False) #去掉None
    for idx in df_dup_na.index:
        smi = df_dup_na.loc[idx, 'InChi']
        same_smi_idx = df[df['InChi'] == smi].index
        df.loc[same_smi_idx, 'check'] = str(list(same_smi_idx))
    return df




if __name__ == '__main__':
    description = 'check structure'
    parser = argparse.ArgumentParser(description)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input csv or sdf directory(file)')
    parser.add_argument('-sminame', default='SMILES',
                        help='smiles column')
    parser.add_argument('-iso', action='store_false', default=True) #立体化学
    parser.add_argument('-o', default=False, help='save new csv file')
    args = parser.parse_args()

    if Path(args.i).suffix == '.csv':
        df = pd.read_csv(args.i)
    elif Path(args.i).suffix == '.sdf':
        df = pt.LoadSDF(args.i)
    if not args.o:
        args.o = args.i

    df_wash = process_molecules(df, args.sminame, args.iso)
    df_wash.to_csv(args.o, index=None)
    # df_wash['ROMol'] = [Chem.MolFromSmiles(i) for i in df_wash['washsmi']]
    # pt.WriteSDF(df, args.o.replace('csv','sdf'), properties=df_wash.columns.to_list(), idName='compound_id')
