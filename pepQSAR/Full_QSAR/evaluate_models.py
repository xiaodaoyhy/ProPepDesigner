import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import glob
import os

def calculate_metrics(y_true, y_pred):
    """Compute metrics after strictly filtering out NaN pairs."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = np.array(y_true)[mask]
    y_pred_clean = np.array(y_pred)[mask]
    
    if len(y_true_clean) == 0:
        return np.nan, np.nan, np.nan
    
    rmse = round(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)), 3)
    mae = round(mean_absolute_error(y_true_clean, y_pred_clean), 3)
    r2 = round(r2_score(y_true_clean, y_pred_clean), 3) 
    return rmse, mae, r2

def extract_model_name(filepath, suffix_to_remove="_predict.csv"):
    basename = os.path.basename(filepath)
    if basename.endswith(suffix_to_remove):
        return basename[:-len(suffix_to_remove)]
    return basename.replace('.csv', '')

def main():
    parser = argparse.ArgumentParser(description="Multi-task model evaluator — wide-format summary matrix")
    parser.add_argument("--csv", type=str, required=True, help="Prediction CSV path(s); wildcards supported")
    parser.add_argument("--model_suffix", type=str, default="_predict.csv", help="Suffix used to extract model name")
    parser.add_argument("--y_cols", nargs="+", required=True, help="Experimental value column names")
    parser.add_argument("--y_pred_cols", nargs="+", required=True, help="Predicted value column names")
    parser.add_argument("--split_col", type=str, required=True, help="Dataset split column (e.g., TrTe)")
    parser.add_argument("--output_csv", type=str, default="model_evaluation_long.csv", help="Output full long-format table")

    args = parser.parse_args()
    
    file_list = glob.glob(args.csv)
    if not file_list:
        raise FileNotFoundError(f"No files matched pattern: {args.csv}")
    
    all_results = []
    # Shorten column names, e.g. mean_pEC50_GLP1R_activity -> GLP1R
    target_mapping = {col: col.replace("mean_pEC50_", "").replace("_activity", "") for col in args.y_cols}
    
    for filepath in file_list:
        model_name = extract_model_name(filepath, args.model_suffix)
        df = pd.read_csv(filepath)
        
        required_cols = args.y_cols + args.y_pred_cols + [args.split_col]
        if any(col not in df.columns for col in required_cols):
            continue
            
        splits = df[args.split_col].dropna().unique()
        
        for split in splits:
            df_split = df[df[args.split_col] == split]
            for y_col, y_pred_col in zip(args.y_cols, args.y_pred_cols):
                y_true = df_split[y_col]
                y_pred = df_split[y_pred_col]
                
                rmse, mae, r2 = calculate_metrics(y_true, y_pred)
                valid_n = (~(np.isnan(y_true) | np.isnan(y_pred))).sum()
                short_target = target_mapping.get(y_col, y_col)
                
                all_results.append({
                    "Model": model_name,
                    "Target": short_target,
                    "Split": split,
                    "Valid_N": valid_n,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2
                })
                
    result_df = pd.DataFrame(all_results)
    # Export full long-format table (includes all Valid_N and MAE)
    result_df.to_csv(args.output_csv, index=False)
    print(f"Detailed long table saved to: {args.output_csv}")
    

    # Keep only RMSE and R2 for horizontal flattening
    pivot_df = result_df.pivot_table(
        index=["Model"], 
        columns=["Target", "Split"], 
        values=["R2", "RMSE"]
    )
    
    # Flatten multi-level columns; capitalize split names (e.g., test -> Test)
    pivot_df.columns = [f"{target}_{str(split).capitalize()}_{metric}" for metric, target, split in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    # Order columns: GIPR -> GLP1R -> GCGR, with Train
    base_cols = ["Model"]
    found_splits = sorted(list(set([c.split('_')[1] for c in pivot_df.columns if c != "Model"])))
    
    ordered_cols = []
    for target in ["GIPR", "GLP1R", "GCGR"]:
        for split in found_splits:
            for metric in ["R2", "RMSE"]:
                col_name = f"{target}_{split}_{metric}"
                if col_name in pivot_df.columns:
                    ordered_cols.append(col_name)
                    
    # Merge and drop any missing columns
    final_cols = base_cols + [c for c in ordered_cols if c in pivot_df.columns]
    pivot_df = pivot_df[final_cols]
    pivot_df_summary = pd.concat([pivot_df,\
                                  pd.DataFrame(data=pivot_df.iloc[:,1:].mean().round(3)).T,\
                                  pd.DataFrame(data=pivot_df.iloc[:,1:].std(ddof=1).round(4)).T])
    
    # Export wide summary table for main text
    pivot_df_summary.to_csv(args.output_csv.replace('long','short'), index=False)
    print(f"Core summary matrix saved to: {args.output_csv.replace('long','short')}")


if __name__ == "__main__":
    main()


# python evaluate_models.py \
#     --csv "./seed_*/TargetData_split_predict_*.csv" \
#     --y_cols mean_pEC50_GIPR_activity mean_pEC50_GLP1R_activity mean_pEC50_GCGR_activity \
#     --y_pred_cols pred_addside_mean_pEC50_GIPR_activity pred_addside_mean_pEC50_GLP1R_activity pred_addside_mean_pEC50_GCGR_activity \
#     --split_col TrTe \
#     --output_csv model_evaluation_long.csv
