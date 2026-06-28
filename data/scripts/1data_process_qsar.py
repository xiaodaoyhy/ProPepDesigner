import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
import os

# Detect outliers
def detect_outliers(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ~group.between(lower_bound, upper_bound)


inpath = '../process/GIPR_GLP1R_GCGR_summary_seq.csv'


activity_name = ['GIPR_activity', 'GLP1R_activity', 'GCGR_activity']
smiles_name = 'washsmi_Iso'
sequence_name = 'pep_sequence'
pep_smiles = 'pep_smiles'
sequence_model_name = 'pep_sequence_model'
pep_model_smiles = 'pep_smiles_model'
length_name = 'length'
check = 'check'
reference = 'reference'	
compound_id = 'compound_name_reference'
note = 'note'
ID = 'ID'

df_init = pd.read_csv(inpath)
outdir = Path(inpath).parent.parent / 'result'

os.makedirs(outdir, exist_ok=True)

# Drop rows that cannot be processed (e.g., cyclic peptides)
df_process1 = df_init[~df_init['pep_sequence'].isna()]
df = df_process1.query(f'{length_name} <= 45').reset_index(drop=True)
print(f'Peptides with length > 45: {len(df_process1)-len(df)}')


summary_data = df.drop_duplicates(subset=[pep_model_smiles]).reset_index(drop=True).loc[:, [ID, smiles_name,sequence_name,\
                                                                                            sequence_model_name,length_name,\
                                                                                            pep_model_smiles,check,\
                                                                                            reference, compound_id, note]]
for activity in activity_name:
    print(f'{activity} data >>>>>>>>>>>>>>>>>>>>>')

    df_single = df[~df[activity].isna()].reset_index(drop=True)
    print(f'Missing values: {len(df)-len(df_single)}')

    df_single[f'{activity}_relation'] = df_single[f'{activity}_relation'].astype(str)
    bad_bounds_mask = df_single[f'{activity}_relation'].str.contains('>', regex=False, na=False) | \
                        df_single[f'{activity}_relation'].str.contains('<', regex=False, na=False) | \
                        df_single[f'{activity}_relation'].str.contains('-', regex=False, na=False)
    df_single.loc[bad_bounds_mask, activity] = np.nan
    df_single = df_single[~df_single[activity].isna()].reset_index(drop=True)
    print(f'Boundary values: {sum(bad_bounds_mask)}')


    # 1. Flag outliers
    df_single[f'pEC50_{activity}'] = df_single[activity].apply(lambda x: -np.log10(x)+9)
    df_single['is_outlier'] = df_single.groupby(pep_model_smiles)[f'pEC50_{activity}'].transform(detect_outliers)
    # 2. Filter out outliers
    non_outlier_data = df_single[~df_single['is_outlier']].copy()
    print(f'Outliers: {len(df_single)-len(non_outlier_data)}')

    # 3. Average activity values per SMILES
    mean_activity = non_outlier_data.groupby(pep_model_smiles)[f'pEC50_{activity}'].mean().reset_index()
    mean_activity.rename(columns={f'pEC50_{activity}': f'mean_pEC50_{activity}'}, inplace=True)
    qs = mean_activity[f'mean_pEC50_{activity}'].quantile([0.5, 0.75, 0.95])
    print(qs)


    # 4. Keep summary statistics
    stats_info = non_outlier_data.groupby(pep_model_smiles)[activity].agg([
        'count', 'std', 'min', 'max'
    ]).reset_index()
    # 5. Merge means and summary statistics
    final_data = pd.merge(mean_activity, stats_info, on=pep_model_smiles)
    # final_data.to_csv(outdir / f'process_{activity}.csv', index=None)
    summary_data = pd.merge(summary_data, mean_activity, on=pep_model_smiles, how='outer')
    print(f'Processed row count: {len(final_data)}\n')

# Drop rows where all three activity values are NaN
summary_data = summary_data[~summary_data[['mean_pEC50_'+i for i in activity_name]].isna().all(axis=1)]
summary_data.to_csv(outdir / 'process_summary1_new.csv', index=None)
print(f'All data processed. Final row count: {len(summary_data)}')
summary_data['aa_count'] = summary_data[sequence_model_name].apply(lambda x: Counter(eval(x.replace('*',''))))
counts = Counter()
summary_data['aa_count'].apply(lambda d: counts.update(d) if isinstance(d, dict) else None)
# print(counts)

# Compute total residue count across all amino acids
total_residuess = sum(counts.values())
print(f"{'Amino Acid':<12} | {'Frequency (%)':<15} | {'Count':<8}")
print("-" * 42)

# Sort by frequency (descending) and print as percentages
for aa, count in counts.most_common():
    frequency = (count / total_residuess) * 100
    print(f"{aa:<12} | {frequency:>13.3f}% | {count:>8}")


