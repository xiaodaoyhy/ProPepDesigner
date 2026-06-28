import os
from typing import List

import pandas as pd
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig
from pepinvent.supervised_learning.trainer.dataset import Dataset
from pepinvent.supervised_learning.utils.log import get_logger
from reinvent_models.mol2mol.models.vocabulary import SMILESTokenizer


class BaseTrainer(ABC):
    def __init__(self, opt: ArchitectureConfig):
        self._config = opt
        self.save_path = os.path.join(self._config.save_directory)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'))
        LOG = get_logger(name="train_model", log_path=os.path.join(self.save_path, 'train_model.log'))
        self.LOG = LOG
        self.LOG.info(self._config)

    def initialize_dataloader(self, data_path, batch_size, vocab, sampler=None):
        data = pd.read_csv(data_path, sep=",")
        dataset = Dataset(data=data, vocabulary=vocab, tokenizer=SMILESTokenizer())
        data_sampler = None
        if sampler is not None:
            # sampler is expected to be a Sampler class (e.g., DistributedSampler)
            # Handle DistributedSampler with shuffle/drop_last explicitly.
            try:
                from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
                if sampler is _DistributedSampler:
                    data_sampler = sampler(
                        dataset,
                        shuffle=self._config.shuffle_each_epoch,
                        drop_last=self._config.drop_last_batch
                    )
                else:
                    data_sampler = sampler(dataset)
            except Exception:
                data_sampler = sampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size,
            drop_last=self._config.drop_last_batch,
            shuffle=(self._config.shuffle_each_epoch and data_sampler is None),
            sampler=data_sampler,
            collate_fn=Dataset.collate_fn,
            pin_memory=True
        )
        return dataloader

    def to_tensorboard(self, train_loss, validation_loss, accuracy, token_accuracy, similarity, epoch, nll_mean,
                       learning_rate=None, **kwargs):
        self.summary_writer.add_scalars("loss", {
            "train": train_loss,
            "validation": validation_loss
        }, epoch)
        self.summary_writer.add_scalar("identity_accuracy/validation", accuracy, epoch)
        self.summary_writer.add_scalar("token accuracy/validation", token_accuracy, epoch)
        self.summary_writer.add_scalar("similarity", similarity, epoch)
        self.summary_writer.add_scalar("-log(likelihood)", nll_mean, epoch)
        if learning_rate is not None:
            self.summary_writer.add_scalar("learning_rate", learning_rate, epoch)
        # Don't close the writer per-epoch; it prevents further logging.
        self.summary_writer.flush()

    @abstractmethod
    def get_model(self, vocab):
        pass

    @abstractmethod
    def get_optimization(self, **kwargs):
        pass

    @abstractmethod
    def train_epoch(self, **kwargs):
        pass

    @abstractmethod
    def validation_stat(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def execute(self, **kwargs):
        pass

