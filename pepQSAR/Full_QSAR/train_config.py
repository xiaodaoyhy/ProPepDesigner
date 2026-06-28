from train import TrainConfigNew, MultiTaskTransformerTrainer
from pathlib import Path
import pandas as pd
import ast


if __name__ == "__main__":

    # Paths and hyperparameters can be adjusted for your environment
    task_name = ['mean_pEC50_GIPR_activity', 'mean_pEC50_GLP1R_activity', 'mean_pEC50_GCGR_activity']
    feature_type = ['ESM', 'RDKit', 'onehot']


    # run_type = 'train'
    # csv_path="../../data/result/TargetData.csv"


    run_type='predict'
    csv_path="../../data/result/filter_full_seq.csv"


    seed_ls = [5,25,39,44,63]

    model_dirs = Path(__file__).resolve().parent


    # ====== Configuration template ======
    # Base config (file paths & task names)
    for seed in seed_ls:
        model_dir = Path(model_dirs) / f'seed_{seed}'
        # csv_path = model_dir / 'TargetData_split.csv'

        base_kwargs = dict(
            run_type=run_type,
            csv_path=csv_path,

            model_dir=model_dir,
            peptide_seq_name='pep_sequence_model',
            peptide_smis_name='pep_smiles_model',
            activity_columns=task_name,
            feature_type=feature_type
        )

        config = TrainConfigNew(
            **base_kwargs,
            seed=seed,
            batch_size=64,
            num_epochs=100,
            split_ratio=0.1,
            split_method='stratify',
            early_stop_patience=10,    # Early stopping patience
            d_model=384,
            nhead=8,
            num_layers=4,
            dropout=0.2,
            lr=1e-4,
            weight_decay=1e-4,
        )


        trainer = MultiTaskTransformerTrainer(config)
        if run_type == 'train':
            trainer.fit()      # Train and save best model
        elif run_type == 'predict':
            # Load new data
            df_new = pd.read_csv(csv_path)
            new_fasta = df_new['pep_sequence_model'].apply(ast.literal_eval).tolist()
            new_frag = df_new['pep_smiles_model'].apply(ast.literal_eval).tolist()

            # Run prediction
            preds_np = trainer.predict(new_fasta, new_frag)

            # Write prediction results
            pred_cols = [f"pred_addside_{name}" for name in config.activity_columns]
            pred_df = pd.DataFrame(data=preds_np, columns=pred_cols)

            # Concatenate with original data by column
            result_df = pd.concat([df_new.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

            # Convert pEC50 back to activity (for columns named like 'pEC50_xxx')
            for col in pred_cols:
                if "pEC50" in col:
                    raw_col = col.replace("pEC50_", "")
                    result_df[raw_col] = (10 ** (9 - result_df[col])).round(4)
            save_cols = ['main_smiles', 'reference', 'ID', 'mean_GIPR_activity', 'pred_GIPR_activity',\
                            'mean_GLP1R_activity', 'pred_GLP1R_activity', 'mean_GCGR_activity', 'pred_GCGR_activity', 'length']
            out_path = Path(model_dir) / Path(csv_path).name.replace(".csv", f"_predict_{seed}.csv")
            result_df.to_csv(out_path, index=False)