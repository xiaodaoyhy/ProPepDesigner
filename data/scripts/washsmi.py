
# -*- coding: UTF-8 -*-


#    python washsmi.py -i test_washsmi.csv

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
        clean_mol = rdMolStandardize.Cleanup(clean_mol) 
        clean_mol = rdMolStandardize.FragmentParent(clean_mol)
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



def process_molecules(df, smi_column, iso):
    if iso:
        washsmi_col = 'washsmi_Iso'
    else:
        washsmi_col = 'washsmi_DelIso'

    df[f'{washsmi_col}'] = df[smi_column].apply(lambda x: washsmi(x, iso))
    df['InChi'] = df[f'{washsmi_col}'].apply(lambda x: washToInChi(x))
    
    
    df['check'] = None
    nan_idx = df[df[f'{washsmi_col}'].isna()].index
    df.loc[nan_idx, 'check'] = -1

    df_dup = df[df.duplicated(subset='InChi', keep='first')] 
    df_dup_na = df_dup.dropna(subset=['InChi'], inplace=False) 
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
    parser.add_argument('-iso', action='store_false', default=True)
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
