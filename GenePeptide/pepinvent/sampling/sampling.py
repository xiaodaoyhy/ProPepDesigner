from typing import List, Dict, Union

import torch.utils.data as tud
import pandas as pd
import torch
import tqdm

from pepinvent.reinforcement.chemistry import Chemistry
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter
from reinvent_models.mol2mol.dataset.dataset import Dataset
from pepinvent.sampling.sampling_config import SamplingConfig
from reinvent_models.mol2mol.enums.sampling_mode_enum import SamplingModesEnum


class Sampling:
    def __init__(self, agent: Union[GenerativeModelBase, Mol2MolAdapter], config: SamplingConfig, ):
        self._agent = agent
        self._config = config
        self._chemistry = Chemistry()
        # Apply optional sampling knobs when available
        if hasattr(self._agent, "set_temperature"):
            self._agent.set_temperature(self._config.temperature)
        if hasattr(self._agent, "set_beam_size"):
            self._agent.set_beam_size(self._config.beam_size)

    def _get_decode_type(self):
        mode = (self._config.decode_type or "multinomial").lower()
        enum = SamplingModesEnum()
        if mode in ("multinomial", "random"):
            return enum.MULTINOMIAL
        if mode in ("greedy",):
            return enum.GREEDY
        if mode in ("beam", "beamsearch"):
            return enum.BEAMSEARCH
        raise ValueError(f"Unknown decode_type '{self._config.decode_type}'. Use: multinomial|greedy|beamsearch")

    def _sample(self, sequences: List[str]) -> Dict[str, List[SampledSequencesDTO]]:
        sequence_dtos = {}
        decode_type = self._get_decode_type()

        # IMPORTANT: building a Dataset/DataLoader per input sequence is slow in Python.
        # We process sequences in chunks and sample in batches for better throughput.
        per_chunk = max(int(getattr(self._config, "sequences_per_chunk", 64)), 1)
        num_workers = max(int(getattr(self._config, "num_workers", 0)), 0)
        show_progress = bool(getattr(self._config, "show_progress", True))

        chunk_starts = list(range(0, len(sequences), per_chunk))
        for chunk_start in tqdm.tqdm(chunk_starts, desc="Sampling (chunks)", disable=not show_progress):
            chunk = sequences[chunk_start:chunk_start + per_chunk]
            input_sequences: List[str] = []
            for s in chunk:
                input_sequences.extend([s] * self._config.num_samples)

            dataset = Dataset(input_sequences, vocabulary=self._agent.vocabulary, tokenizer=self._agent.tokenizer)
            data_loader = tud.DataLoader(
                dataset,
                self._config.batch_size,
                shuffle=False,
                collate_fn=Dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(num_workers > 0),
            )

            for batch in tqdm.tqdm(data_loader, desc="Sampling (batches)", leave=False, disable=not show_progress):
                src, src_mask = batch
                dtos = self._agent.sample(src, src_mask, decode_type=decode_type)
                for dto in dtos:
                    sequence_dtos.setdefault(dto.input, []).append(dto)
        return sequence_dtos

    def execute(self):
        masked_peptides = self.load_data()
        dtos = self._sample(masked_peptides)
        report = self._create_report(dtos)
        self._save_reports(report)

    def load_data(self) -> List[str]:
        test_data = pd.read_csv(self._config.input_sequences_path)
        masked_peptides = list(test_data['Source_Mol'])
        return masked_peptides

    def _create_report(self, results: Dict[str, List[SampledSequencesDTO]]) -> pd.DataFrame:
        rows = []
        max_n = 0
        for key, dtos in results.items():
            outputs = [dto.output for dto in dtos]
            nlls = [dto.nll for dto in dtos]
            max_n = max(max_n, len(outputs))
            rows.append({"Input": key, "Output": outputs, "NLLs": nlls})

        dataframe = pd.DataFrame(rows)
        if len(rows) == 0:
            return pd.DataFrame(columns=["Input", "NLLs"])

        column_labels = [f"Generated_smi_{x}" for x in range(1, max_n + 1)]
        split = pd.DataFrame(dataframe["Output"].to_list(), columns=column_labels)
        dataframe = pd.concat([dataframe.drop(columns=["Output"]), split], axis=1)
        return dataframe

    def _save_reports(self, output: pd.DataFrame):
        output.to_csv(self._config.results_output)
