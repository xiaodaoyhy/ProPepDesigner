import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from typing import List, Tuple

from rdkit import RDLogger

# 完全禁用RDKit的所有警告
RDLogger.DisableLog('rdApp.*')

# 天然氨基酸 SMILES 字符串字典
natural_aa_smiles ={'R': 'N=C(N)NCCC[C@H](N)C(=O)O',
 'H': 'N[C@@H](Cc1c[nH]cn1)C(=O)O',
 'K': 'NCCCC[C@H](N)C(=O)O',
 'D': 'N[C@@H](CC(=O)O)C(=O)O',
 'E': 'N[C@@H](CCC(=O)O)C(=O)O',
 'S': 'N[C@@H](CO)C(=O)O',
 'T': 'C[C@@H](O)[C@H](N)C(=O)O',
 'N': 'NC(=O)C[C@H](N)C(=O)O',
 'Q': 'NC(=O)CC[C@H](N)C(=O)O',
 'C': 'N[C@@H](CS)C(=O)O',
 'G': 'NCC(=O)O',
 'A': 'C[C@H](N)C(=O)O',
 'P': 'O=C(O)[C@@H]1CCCN1',
 'I': 'CC[C@H](C)[C@H](N)C(=O)O',
 'L': 'CC(C)C[C@H](N)C(=O)O',
 'M': 'CSCC[C@H](N)C(=O)O',
 'F': 'N[C@@H](Cc1ccccc1)C(=O)O',
 'W': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
 'Y': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',
 'V': 'CC(C)[C@H](N)C(=O)O'}


def calculate_all_descriptors(smiles_dict):
    """为字典中的每个SMILES计算所有可用的描述符。"""
    mols = {name: Chem.MolFromSmiles(smiles) for name, smiles in smiles_dict.items()}
    # 移除无法解析的分子
    mols = {name: mol for name, mol in mols.items() if mol is not None}
    # 存储所有描述符数据的 DataFrame
    all_results = []
    
    for name, mol in mols.items():
        # 使用 CalcMolDescriptors 计算所有描述符
        desc_values = Descriptors.CalcMolDescriptors(mol)
        desc_values['Name'] = name  # 添加名称列
        all_results.append(desc_values)
    
    # 创建DataFrame
    df_all_desc = pd.DataFrame(all_results)
    df_all_desc.set_index('Name', inplace=True)
    
    return df_all_desc


def _prepare_natural_reference(method: str = "cosine"):
    """
    预计算天然氨基酸描述符参考矩阵（只计算一次）。

    返回:
        natural_names: pd.Index
        valid_cols: List[str]
        scaler: StandardScaler
        scaled_natural: np.ndarray, shape [n_natural, n_features]
    """
    if method not in {"euclidean", "cosine"}:
        raise ValueError("method 必须是 'euclidean' 或 'cosine'")

    df_natural = calculate_all_descriptors(natural_aa_smiles)

    # 移除包含 NaN/Inf 的列（天然参考端）
    valid_cols = []
    for col in df_natural.columns:
        if df_natural[col].isna().any() or np.isinf(df_natural[col]).any():
            continue
        valid_cols.append(col)

    df_natural = df_natural[valid_cols]

    scaler = StandardScaler()
    scaled_natural = scaler.fit_transform(df_natural.values)
    natural_names = df_natural.index

    return natural_names, valid_cols, scaler, scaled_natural


# 模块加载时预先准备天然AA参考（避免每次调用重复计算）
_NATURAL_NAMES, _VALID_COLS, _SCALER, _SCALED_NATURAL = _prepare_natural_reference(method="cosine")

def map_single_smiles_to_aa(smiles: str, topk: int = 1):
    """
    输入单个氨基酸的 SMILES，返回最相似的天然氨基酸单字母。

    说明：
    - 这里使用 RDKit 的 CalcMolDescriptors 生成一组分子描述符，
      然后以“天然氨基酸（单体）”作为参考集合做相似性匹配。
    - 为了效率，天然氨基酸参考描述符会在模块加载时预先计算一次。

    参数:
        smiles: str
            非天然氨基酸的SMILES

        topk: int
            返回 topk 个候选（包含分数），便于你检查映射是否合理

    返回:
        best_match: str
            最匹配的天然氨基酸单字母
        topk_matches: List[Tuple[str, float]]
            topk 候选列表 [(aa, score), ...]
            - method='cosine' 时 score 为相似度（越大越好）
    """

    mol = Chem.MolFromSmiles(smiles)

    # 计算 query 描述符，并对齐到 valid_cols
    desc = Descriptors.CalcMolDescriptors(mol)
    x = np.array([desc[c] for c in _VALID_COLS], dtype=float).reshape(1, -1)

    # 标准化到天然AA分布
    x_scaled = _SCALER.transform(x)[0]

    scores: List[Tuple[str, float]] = []
    for j, aa in enumerate(_NATURAL_NAMES):
        nat_vec = _SCALED_NATURAL[j]
        score = float(1.0 - cosine(x_scaled, nat_vec))
        scores.append((aa, score))


    scores.sort(key=lambda t: t[1], reverse=True)  # 相似度大优先
    best_match = scores[0][0]
    return best_match, scores[: max(1, int(topk))]




if __name__ == "__main__":

    test_smiles = "N[C@](CC(C)C)(C)C(=O)O"  # V 的单体 SMILES
    best, top = map_single_smiles_to_aa(test_smiles, topk=1)
    best = map_single_smiles_to_aa(test_smiles, topk=1)

    print("Best:", best)
    print("Top3:", top)
