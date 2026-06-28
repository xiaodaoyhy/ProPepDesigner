from train import TrainConfigNew, MultiTaskMLPTrainer
from pathlib import Path
import pandas as pd
import ast


if __name__ == "__main__":

    # Paths and hyperparameters can be adjusted for your environment
    task_name = ['mean_pEC50_GIPR_activity', 'mean_pEC50_GLP1R_activity', 'mean_pEC50_GCGR_activity']

    # ['PhysChem', 'RDKit', 'ESM', 'NNAA', 'ECFP4', 'CTD']
    feature_type = ['ESM', 'RDKit', 'NNAA']

    
    seed_ls = [7,27,29,39,40]
    # seed_ls = [40]

    model_dirs = Path(__file__).resolve().parent
    run_type = 'train'
    csv_path="../../data/result/TargetData_removeLongside.csv"

    # run_type='predict'


    # temp(csv_path, task_name)
    for seed in seed_ls:
        model_dir = Path(model_dirs) / f'seed_{seed}'
        # csv_path = model_dir / 'modeling_data.csv'

        # ====== Configuration template ======
        # Base config (file paths & task names)
        base_kwargs = dict(
            run_type=run_type,
            csv_path=csv_path,
            model_dir=model_dir,
            peptide_seq_ls_name='pep_sequence_model',
            peptide_smis_ls_name='pep_smiles_model',
            main_smis_name='Backbone_smiles',

            activity_columns=task_name,
            feature_type=feature_type,
            split_ratio=0.1,
            split_method='stratify',
        )


        # Model hyperparameters
        config = TrainConfigNew(
            **base_kwargs,
            seed=seed,
            batch_size=64,
            num_epochs=200,
            early_stop_patience=10,    # Early stopping patience
            hidden=[512, 128],
            # hidden=[256, 128],
            # hidden=[64, 32],

            dropout=0.2,
            lr=1e-4,
            weight_decay=1e-4,
            task_weight_strategy="inverse_freq"  # Auto-compute weights from train-set valid label counts

        )

        trainer = MultiTaskMLPTrainer(config)
        if run_type == 'train':
            trainer.fit()      # Train and save best model
        elif run_type == 'predict':
            # Load new data
            df_new = pd.read_csv(csv_path)
            new_fasta = df_new['pep_sequence_model'].apply(ast.literal_eval).tolist()
            new_frag = df_new['pep_smiles_model'].apply(ast.literal_eval).tolist()
            new_smiles = df_new['Backbone_smiles'].tolist()


            # Run prediction
            predict_results = trainer.predict(new_fasta, new_frag, new_smiles)

            # Extract prediction matrix [N, 3] and build base prediction columns
            pred_cols = [f"pred_{name}" for name in config.activity_columns]
            pred_df = pd.DataFrame(data=predict_results["predictions"], columns=pred_cols)

            # Append applicability domain (AD) metrics
            pred_df["AD_leverage"] = predict_results["leverages"]     # Leverage per molecule
            pred_df["is_inside_AD"] = predict_results["is_inside_AD"]  # In-domain flag (True/False)


            # Concatenate with original data by column
            result_df = pd.concat([df_new.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            # Convert pEC50 back to activity (for columns named like 'pEC50_xxx')
            for col in pred_cols:
                if "pEC50" in col:
                    raw_col = col.replace("pEC50_", "")
                    result_df[raw_col] = (10 ** (9 - result_df[col])).round(4)
            save_cols = ['Backbone_smiles', 'reference', 'ID', 'mean_GIPR_activity', 'pred_GIPR_activity',\
                            'mean_GLP1R_activity', 'pred_GLP1R_activity', 'mean_GCGR_activity', 'pred_GCGR_activity', 'length']

            out_path = Path(model_dir) / Path(csv_path).name.replace(".csv", f"_predict_{seed}.csv")
            result_df.to_csv(out_path, index=False)