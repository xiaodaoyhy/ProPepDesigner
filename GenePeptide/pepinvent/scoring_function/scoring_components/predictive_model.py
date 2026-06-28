
from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters

from pathlib import Path
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Optional, List, Union, Dict, Tuple
import pandas as pd
from rdkit import RDLogger
# 禁用所有rdApp.*相关的警告
RDLogger.DisableLog('rdApp.*')

import numpy as np
import joblib
from myutils.smiles2seq import peptide_smiles2seq
from pepQSAR.Backbone_QSAR.model_base import MultiTaskMLP
from pepQSAR.Backbone_QSAR.embedding import getEmbedding



class PredictiveModel(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self.model_path = Path(self.parameters.specific_parameters.get('model_path'))
        self.AD_path = Path(self.parameters.specific_parameters.get('model_path')) / "model_AD.pkl"


        self.structure = joblib.load(self.model_path  / 'model_structure.pkl')
        self._model = self._load_model(self.model_path  / 'model.pth')
        self.label_scaler = np.load(Path(self.model_path / "label_scaler.npz"))

        self._last_outputs = None   # 缓存上次的 forward 结果
        self._last_key = None       # 缓存上次的输入哈希
   

    def _transformation(self, scores: np.ndarray) -> np.ndarray:
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores)

    def _smiles2seq(self, smiles):
        res = peptide_smiles2seq(smiles=smiles)
        if res is None:
            return None, None
        single_fasta_list, single_frags_list, _, _ = res
        return single_fasta_list, single_frags_list

    def _load_model(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiTaskMLP(
            input_dim=self.structure['input_dim'],
            output_dim=self.structure['output_dim'],
            hidden_layers=self.structure['hidden'],
            dropout=self.structure['dropout']
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    

    def _forward_once(self, molecules: ScoringInputDTO) -> Dict[str, np.ndarray]:
        peptides: List[str] = list(molecules.peptides or [])
        n_total = len(peptides)

        # 没有输入：直接返回空
        if n_total == 0:
            return {
                "predictions": np.zeros((0, int(self.structure["output_dim"])), dtype=np.float32),
                "ad_valid_mask": np.zeros((0,), dtype=bool),
                "ad_embeddings_valid": np.zeros((0, 0), dtype=np.float32),
            }

        valid_indices: List[int] = []
        fasta_valid: List = []
        frags_valid: List = []
        smiles_valid: List[str] = []


        for i, pep in enumerate(peptides):
            single_fasta, single_frags = self._smiles2seq(pep)
            valid_indices.append(i)
            fasta_valid.append(single_fasta)
            frags_valid.append(single_frags)
            smiles_valid.append(pep)
            if i % 10 == 0:
                print(single_fasta)
        output_dim = int(self.structure["output_dim"])


        predictions_full = np.full((n_total, output_dim), -1.0e6, dtype=np.float32)
        ad_valid_mask = np.zeros((n_total,), dtype=bool)
        ad_embeddings_valid: np.ndarray

        if len(valid_indices) == 0:
            return {
                "predictions": predictions_full,
                "ad_valid_mask": ad_valid_mask,
                "ad_embeddings_valid": np.zeros((0, 0), dtype=np.float32),
            }


        ad_valid_mask[np.array(valid_indices, dtype=int)] = True

        input_dataset = getEmbedding(
            fasta_list=fasta_valid,
            frag_list=frags_valid,
            main_smiles_list=smiles_valid,
            feature_type=self.structure["feature_type"],
            run_type="predict",
            save_path=self.model_path,
        )
        dataloader = DataLoader(input_dataset, batch_size=32)

        # AD 用的理化特征,返回 numpy
        ad_X = input_dataset.tensors[0].detach().cpu().numpy()

        device = next(self._model.parameters()).device
        y_pred = []
        with torch.no_grad():
            for batch_embeddings, _, _ in dataloader:
                outputs = self._model(batch_embeddings.to(device))
                y_pred.append(outputs.cpu())

        predictions_valid = torch.cat(y_pred).numpy()
        predictions_valid = predictions_valid * self.label_scaler["std"] + self.label_scaler["mean"]

        # 回填到原 batch 顺序
        for local_i, global_i in enumerate(valid_indices):
            predictions_full[global_i, :] = predictions_valid[local_i, :]

        return {
            "predictions": predictions_full,
            "ad_valid_mask": ad_valid_mask,
            "ad_embeddings_valid": ad_X,
        }
    

    def calculate_score(self, molecules: ScoringInputDTO, step=-1) -> ComponentSummary:
        # === 生成输入 key（根据 peptide 列表） ===
        mol_key = hashlib.md5("".join(molecules.peptides).encode()).hexdigest()

        # === 如果输入不同，重新 forward ===
        if self._last_key != mol_key:
            self._last_outputs = self._forward_once(molecules)
            self._last_key = mol_key
      
        # === 根据子类的 target index 取对应通道 ===
        raw_scores = self._last_outputs["predictions"][:, self._target_index]
        scores = self._transformation(self._last_outputs["predictions"][:, self._target_index])

        # print(f'模型{self._target_index}预测值为{scores}')
        return ComponentSummary(
            total_score=scores,
            parameters=self.parameters,
            raw_score=raw_scores
        )


class PredictiveModel_GIPR(PredictiveModel):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self._target_index = 0

class PredictiveModel_GLP1R(PredictiveModel):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self._target_index = 1

class PredictiveModel_GCGR(PredictiveModel):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        self._target_index = 2

    
class Model_AD(PredictiveModel):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)
        ad_cfg = joblib.load(self.AD_path)
        self.pca_model = ad_cfg['pca_model']
        self.n_train = ad_cfg['n_train']
        self.xtx_inv = ad_cfg['xtx_inv']
        self.h_star = ad_cfg['h_star']

    def calculate_score(self, molecules: ScoringInputDTO, step=-1) -> ComponentSummary:
        # 1. 确保缓存已更新
        mol_key = hashlib.md5("".join(molecules.peptides).encode()).hexdigest()
        if self._last_key != mol_key:
            self._last_outputs = self._forward_once(molecules)
            self._last_key = mol_key

        n_total = len(molecules.peptides or [])
        valid_mask = self._last_outputs.get("ad_valid_mask", np.ones((n_total,), dtype=bool))

        # 2) 对有效样本计算精确的 leverage
        leverages = np.full((n_total,), np.inf, dtype=np.float64)
        if n_total > 0 and np.any(valid_mask):
            x_scaled = self._last_outputs.get("ad_embeddings_valid", None)
            
            # 先投影到 PCA 空间，将维度从 349 维降到 p_reduced 维
            x_pca = self.pca_model.transform(x_scaled)
            
            # 在 PCA 空间计算马氏距离，并加上截距项 1/n
            mahalanobis_part = np.sum((x_pca @ self.xtx_inv) * x_pca, axis=1)
            leverages_valid = (1.0 / self.n_train) + mahalanobis_part
            
            leverages[np.where(valid_mask)[0]] = leverages_valid

        
        # 保留平滑的惩罚：
        # 当 leverages 越大于 h_star 时，得分呈指数级衰减
        ad_scores = np.array(
            [1.0 if l <= self.h_star else math.exp(-(l - self.h_star) * 10.0) for l in leverages]
        )

        return ComponentSummary(
            total_score=ad_scores,
            parameters=self.parameters,
            raw_score=leverages
        )