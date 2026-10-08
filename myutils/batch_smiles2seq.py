from rdkit import RDLogger
# Disable all rdApp.* related warnings
RDLogger.DisableLog('rdApp.*')
from myutils.smiles2seq import peptide_smiles2seq

import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import lru_cache



@lru_cache(maxsize=None)
def smiles2seq_cached(smiles: str):
    seq_ls, seq_smiles, seq_ls_raw, seq_smiles_raw = peptide_smiles2seq(smiles)
    pep_sequence_model = str(seq_ls)
    pep_sequence_smiles = str(seq_smiles)
    pep_sequence = '--'.join(seq_ls_raw)
    pep_smiles = str(seq_smiles_raw)
    length = len(seq_ls)
    nnaa_items = tuple((smi, seq) for seq, smi in zip(seq_ls_raw, seq_smiles_raw) if '*' in seq)
    return pep_sequence, pep_smiles, length, nnaa_items, pep_sequence_model, pep_sequence_smiles



# def _worker(smiles: str):
#     return smiles, smiles2seq_cached(smiles)

def _worker(smiles: str):
    try:
        return smiles, smiles2seq_cached(smiles)
    except RecursionError:
        print(f"\n[RecursionError] SMILES causing infinite loop or excessive recursion: {smiles}")
        # Return empty values matching the normal result format so the subsequent pd.DataFrame does not error
        return smiles, (None, None, 0, tuple(), None, None)
    except Exception as e:
        print(f"\n[Other Error] Parsing failed: {e} | SMILES: {smiles}")
        return smiles, (None, None, 0, tuple(), None, None)


def mp_smiles2seq(df, smiles_column):

    # 1) Get the SMILES list
    ls_smis = df[smiles_column].tolist()
    # Use multiprocessing for parallel processing
    with mp.Pool(mp.cpu_count()-8) as pool:
        results = dict(pool.imap_unordered(_worker, ls_smis, chunksize=200))

    # 2) Map results back to the DataFrame in one step
    mapped = df[smiles_column].map(results)
    cols = ['pep_sequence', 'pep_smiles', 'length', '_nnaa_items', 'pep_sequence_model','pep_smiles_model']
    result_df = pd.DataFrame.from_records(mapped.tolist(), columns=cols, index=df.index)
    df[cols] = result_df

    # 3) Aggregate NNAA (merge once instead of repeated updates in a loop)
    from collections import ChainMap
    NNAA = dict(ChainMap(*[dict(x) for x in df['_nnaa_items'] if x]))
    df.drop(columns=['_nnaa_items'], inplace=True)
    return df