from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from itertools import chain
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from multiprocessing import Pool
from myutils.batch_smiles2seq import mp_smiles2seq


# ==============================================
# ECFP4核心计算函数 (确保输出 BitVect),计算相似性
# ==============================================
def smiles_to_fp_object(smiles):
    """返回 RDKit 原生 BitVect 对象，用于 Tanimoto 计算"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    except:
        return None
    return None

def process_list(smi_list):
    """并行处理 SMILES 列表"""
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(smiles_to_fp_object, smi_list), 
                            total=len(smi_list), desc="Processing"))
    return results



def calculate_sequence_identity(seq1, seq2) -> float:
    """
    计算两条等长、已对齐多肽序列之间的一致性
    """
    print(seq1, seq2)
    print(len(seq1), len(seq2))
    if not seq1 or not seq2:
        return 0.0
    assert len(seq1) == len(seq2), "Peptide sequences must be of equal length."
    
    total_alignment_length = len(seq1)
    identical_residues = sum(pos1 == pos2 for pos1, pos2 in zip(seq1, seq2))
    return round(identical_residues / total_alignment_length, 2)



# 针对一个掩码序列
def evaluate_one_mask_row(input_smi: str, generated_smis: list, row_id: int):
    """
    对单条掩码输入（Input）对应的多次采样（Generated_smi_*）做评价，并返回：
    - summary: dict（该 Input 的汇总指标）
    - df_pepsmiles_unique: 去重后的分子级结果（含序列、相似性等）
    """
    input_pep = str(input_smi).split("|")
    masking_positions = [i for i, s in enumerate(input_pep) if s == "?"] # 掩码位置
    pep_len = len(input_pep) # 多肽长度
  
    df_init = pd.DataFrame({"Generated_smi": generated_smis})
    df_init = df_init.dropna(subset=["Generated_smi"]).reset_index(drop=True)
    df_init["Generated_smi"] = df_init["Generated_smi"].astype(str)

    # 基本信息
    df_init["mask_length"] = df_init["Generated_smi"].apply(lambda x: x.count("|") + 1)
    print(f"\n[row={row_id}] 多肽长度为：{pep_len}")
    print(f"[row={row_id}] 掩码位置为：{masking_positions}, 共{len(masking_positions)}位置")
    print(f"[row={row_id}] 采样数量为：{len(df_init)}")


    # 1、长度有效性：掩码长度与采样得到的长度一致
    df_len_valid = df_init[df_init["mask_length"] == len(masking_positions)].reset_index(drop=True)
    len_valid = len(df_len_valid)
    len_valid_ratio = len_valid / len(df_init) if len(df_init) else 0.0
    print(
        f"[row={row_id}] 采样数量为{len(df_init)},长度无效数量：{len(df_init) - len_valid},长度有效比例：{len_valid_ratio}"
    )

    
    # 2、结构有效性:基于长度有效的多肽
    # 输入骨架（从 Input 还原），用于把掩码 AA 填回去
    def padding_mask(masking_positions:list, gene_smis: list, refer_pep: list):
        for i,gene_smi in zip(masking_positions, gene_smis):
            refer_pep[i] = gene_smi
        return refer_pep

    df_len_valid['Generated_smi_ls'] = df_len_valid['Generated_smi'].apply(lambda x: x.split('|'))
    df_len_valid['pep_smiles_temp_ls'] = df_len_valid['Generated_smi_ls'].apply(lambda x: padding_mask(masking_positions, x, input_pep.copy()))
    df_len_valid['smiles'] = df_len_valid['pep_smiles_temp_ls'].apply(lambda x: ''.join(x))
    del df_len_valid['pep_smiles_temp_ls']
    del df_len_valid['Generated_smi_ls']

    df_len_valid = mp_smiles2seq(df_len_valid, 'smiles') #更新了pep_smiles_model和length列
    df_structure_valid = df_len_valid.query(f'length == {len(input_pep)} and pep_smiles.notna()')
    structure_valid_ratio = len(df_structure_valid)/len(df_len_valid) if len(df_len_valid)!=0 else 0
    print(f"{len(df_len_valid)}个结果中的有效结构为：{len(df_structure_valid)}, 结构有效性为：{structure_valid_ratio}")


    # 3. 多肽唯一性
    df_structure_uniq = df_structure_valid.drop_duplicates(subset=['pep_smiles'], keep='first').copy()
    df_structure_uniq.reset_index(drop=True, inplace=True)  # 使用 drop=True 避免保留旧索引
    pep_unique_ratio = len(df_structure_uniq) / len(df_structure_valid) if len(df_structure_valid)!=0 else 0
    print(f"[row={row_id}] {len(df_structure_valid)}个结果中的peptide唯一性为：{pep_unique_ratio}")


    # 4、生成的天然氨基酸数量 & 非氨基酸数量 & 新颖性统计
    df_structure_uniq.loc[:, 'Generated_aa_smis'] = df_structure_uniq["pep_smiles_model"].apply(lambda x: [eval(x)[i] for i in masking_positions])
    aa_type = list(chain.from_iterable(df_structure_uniq["Generated_aa_smis"].values.tolist())) if len(df_structure_uniq) else []
    aa_unique1 = len(set(aa_type))
    aa_unique2 = len(
        set([Chem.MolToSmiles(Chem.MolFromSmiles(i), isomericSmiles=False) for i in aa_type])) if aa_type else 0
    
    aa_natural, aa_nnaa, aa_new = 0, 0, 0
    for smi in set(aa_type):
        aa_name = aa_db.get(smi, "X")
        if aa_name in "ACDEFGHIKLMNPQRSTVWY" and len(aa_name)==1:
            aa_natural += 1
        elif aa_name == "X":
            aa_new += 1
        else:
            aa_nnaa += 1
    print(f'生成的天然氨基酸数量为：{aa_natural}个')
    print(f'生成的非天然氨基酸数量为：{aa_nnaa}个')
    print(f'生成的新的氨基酸数量为：{aa_new}个')

    summary = {
        "row_id": row_id,
        "Input": input_smi,
        "mask_positions": str(masking_positions),
        "n_mask": len(masking_positions),
        "n_sample": len(df_init),
        "n_len_valid": len_valid,
        "n_structure_valid": len(df_structure_valid),
        "len_valid_ratio": len_valid_ratio,
        "structure_valid_ratio": structure_valid_ratio,
        "pep_unique_ratio": pep_unique_ratio,
        "aa_unique_isomeric": aa_unique1,
        "aa_unique_canonical": aa_unique2,
        "aa_natural": aa_natural,
        "aa_nnaa": aa_nnaa,
        "aa_new": aa_new,
        "aa_type": list(set(aa_type)),
        "n_unique_peptides": len(df_structure_uniq),
    }
    return summary, df_structure_uniq



# ==============================================
# 输入
# ==============================================
# inpath = '../GenePeptide/model/finetune-use/sampling/targetTest_finetune_model_2_46_output.csv'
inpath = '/home/yinhongyan/github/ProPepDesigner_new/GenePeptide/model/base/sampling/targetTest_base_model42_output.csv'

print(Path(inpath).parent)
outdir = Path(inpath).parent

aa_db_init = json.load(open('../data/result/base_trainset_new_all_aa_dict.json', 'r'))
aa_db = {value: key for key, value in aa_db_init.items()}

end_mask = False # 是否在C端进行掩码
sampling_num = 500 # 采样数量




if __name__ == "__main__":
    # ===== 读取采样文件：支持“多条 Input（多条掩码序列）”的汇总分析 =====
    df_sampling = pd.read_csv(inpath)
    gen_cols = [c for c in df_sampling.columns if c.startswith("Generated_smi_")]
    if sampling_num is not None:
        gen_cols = [
            c for c in gen_cols
            if c.split("_")[-1].isdigit() and int(c.split("_")[-1]) <= sampling_num
        ]
    gen_cols = sorted(gen_cols, key=lambda x: int(x.split("_")[-1]))

    all_summaries = []
    all_unique = []

    for row_id, row in df_sampling.iterrows():
        input_smi = row.get("Input")
        generated_smis = [row.get(c) for c in gen_cols]
        summary, df_unique = evaluate_one_mask_row(input_smi, generated_smis, row_id=row_id)
        all_summaries.append(summary)
        if len(df_unique) > 0:
            df_unique.insert(0, "row_id", row_id)
            df_unique.insert(1, "Input", input_smi)
            all_unique.append(df_unique)

    df_summary = pd.DataFrame(all_summaries)
    summary_path = str(outdir / (Path(inpath).stem + "_summary_by_input.csv"))
    df_summary.to_csv(summary_path, index=False)
    print(f"已保存每条 Input 的汇总统计：{summary_path}")

    if all_unique:
        df_all_unique = pd.concat(all_unique, ignore_index=True)
        unique_path = str(outdir / (Path(inpath).stem + "_all_unique.csv"))
        df_all_unique.to_csv(unique_path, index=False)
        print(f"已保存所有 Input 合并后的去重结果：{unique_path}")

    # ===== 总体汇总表（overall） =====
    # 说明：
    # - 采用 macro：先对每条 Input 计算比例，再对这些比例取均值/中位数（每条 Input 等权）。
    overall = {
        "n_inputs": int(len(df_summary)),
        "total_samples": int(df_summary["n_sample"].sum()) if "n_sample" in df_summary.columns else 0,
        "total_len_valid": int(df_summary["n_len_valid"].sum()) if "n_len_valid" in df_summary.columns else 0,
        "total_structure_valid": int(df_summary["n_structure_valid"].sum()) if "n_structure_valid" in df_summary.columns else 0,
        "len_valid_ratio_macro_mean": float(df_summary["len_valid_ratio"].mean()) if len(df_summary) else 0.0,
        "len_valid_ratio_macro_median": float(df_summary["len_valid_ratio"].median()) if len(df_summary) else 0.0,
        "structure_valid_ratio_macro_mean": float(df_summary["structure_valid_ratio"].mean()) if len(df_summary) else 0.0,
        "structure_valid_ratio_macro_median": float(df_summary["structure_valid_ratio"].median()) if len(df_summary) else 0.0,
        "pep_unique_ratio_macro_mean": float(df_summary["pep_unique_ratio"].mean()) if len(df_summary) else 0.0,
        "pep_unique_ratio_macro_median": float(df_summary["pep_unique_ratio"].median()) if len(df_summary) else 0.0,
        # 逐 Input 的 unique 数求和（注意：这不是全局去重后的 unique 数）
        "total_unique_peptides_by_input_sum": int(df_summary["n_unique_peptides"].sum()) if "n_unique_peptides" in df_summary.columns else 0,
    }

    # 全局唯一（跨 Input 去重）
    if all_unique:
        overall["total_unique_rows_concat"] = int(len(df_all_unique))
        overall["global_unique_peptides_smiles_nunique"] = int(df_all_unique["pep_smiles"].nunique())

        # 对每个 target 的 seq_identity 做总体统计（在 df_all_unique 上统计）
        identity_cols = [c for c in df_all_unique.columns if c.endswith("_seq_identity")]
        for c in identity_cols:
            overall[f"{c}_mean"] = float(df_all_unique[c].mean())
            overall[f"{c}_median"] = float(df_all_unique[c].median())
            overall[f"{c}_max"] = float(df_all_unique[c].max())
    else:
        overall["total_unique_rows_concat"] = 0
        overall["global_unique_peptides_smiles_nunique"] = 0

    overall_path = str(outdir / (Path(inpath).stem + "_summary_overall.csv"))
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    print(f"已保存总体汇总统计：{overall_path}")