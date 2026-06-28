from rdkit import RDLogger
# 禁用所有rdApp.*相关的警告
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
        print(f"\n[RecursionError] 发现导致死循环或超深递归的 SMILES: {smiles}")
        # 返回一组与正常结果格式一致的空值，确保后面的 pd.DataFrame 不会报错
        return smiles, (None, None, 0, tuple(), None, None)
    except Exception as e:
        print(f"\n[Other Error] 解析失败: {e} | SMILES: {smiles}")
        return smiles, (None, None, 0, tuple(), None, None)


def mp_smiles2seq(df, smiles_column):

    # 1) 切割计算
    ls_smis = df[smiles_column].tolist()
    # 使用多进程处理
    with mp.Pool(mp.cpu_count()-8) as pool:
        results = dict(pool.imap_unordered(_worker, ls_smis, chunksize=200))

    # 2) 一次性把结果映射回 DataFrame
    mapped = df[smiles_column].map(results)
    cols = ['pep_sequence', 'pep_smiles', 'length', '_nnaa_items', 'pep_sequence_model','pep_smiles_model']
    result_df = pd.DataFrame.from_records(mapped.tolist(), columns=cols, index=df.index)
    df[cols] = result_df

    # 3) 汇总 NNAA（一次合并，替代循环里反复 update）
    from collections import ChainMap
    NNAA = dict(ChainMap(*[dict(x) for x in df['_nnaa_items'] if x]))
    df.drop(columns=['_nnaa_items'], inplace=True)
    return df