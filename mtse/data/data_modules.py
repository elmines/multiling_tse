# STL
from __future__ import annotations
import pathlib
# 3rd Party
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Sampler
import lightning as L
from tqdm import tqdm
from typing import Tuple, List, Tuple, Sequence
# Local
from .encoder import Encoder, PredictTask
from .dataset import MapDataset
from .corpus import StanceCorpus
from .parse import DetCorpusType, CORPUS_PARSERS
from ..constants import DEFAULT_BATCH_SIZE, UNRELATED_TARGET

class BaseDataModule(L.LightningDataModule):
    """
    Dummy placeholder, just to constrain what classes the CLI permits
    """

class PredictDataModule(BaseDataModule):
    def __init__(self, corpora: List[StanceCorpus], batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.encoder: Encoder = None
        self.batch_size = batch_size
        self.corpora = corpora
        self.__datasets: List[Dataset] = []

    def setup(self, stage):
        if self.__datasets:
            return
        for corpus in self.corpora:
            samples = list(corpus)
            encode_iter = tqdm(map(lambda s: self.encoder.encode(s, inference=True), samples), desc=f"Encoding {corpus}")
            self.__datasets.append(MapDataset(encode_iter))

    def predict_dataloader(self):
        return [DataLoader(ds, batch_size=self.batch_size, collate_fn=self.encoder.collate) for ds in self.__datasets]
    def test_dataloader(self):
        return self.predict_dataloader()

class TaskSampler(Sampler):
    def __init__(self,
                 stance_indices: np.ndarray,
                 target_indices: np.ndarray,
                 batch_size: int):
        self.stance_indices = stance_indices
        self.target_indices = target_indices
        self.batch_size = batch_size

        stance_len = len(self.stance_indices)
        self.__n_stance_batches = stance_len // batch_size + bool(stance_len % batch_size)
        target_len = len(self.target_indices)
        self.__n_target_batches = target_len // batch_size + bool(target_len % batch_size)

    def __len__(self):
        return self.__n_stance_batches + self.__n_target_batches

    def __iter__(self):
        permuted_stance_inds = np.random.permutation(self.stance_indices)
        permuted_target_inds = np.random.permutation(self.target_indices)
        mixed_batches = np.array_split(permuted_stance_inds, self.__n_stance_batches) + np.array_split(permuted_target_inds, self.__n_target_batches)
        random.shuffle(mixed_batches)
        mixed_batches = [torch.tensor(inds) for inds in mixed_batches]
        return iter(mixed_batches)

class MixedTrainingDataModule(BaseDataModule):
    def __init__(self,
                 stance_train_corpus: StanceCorpus,
                 target_train_corpus: StanceCorpus,
                 val_corpus: StanceCorpus,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.encoder: Encoder = None

        self.stance_train_corpus = stance_train_corpus
        self.target_train_corpus = target_train_corpus
        self.val_corpus = val_corpus
        self.batch_size = batch_size

        self.__train_ds: Dataset = None
        self.__n_stance: int = None
        self.__val_ds: Dataset = None

    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__n_stance is not None:
            return
        permitted_stances = {'favor', 'against'}
        raw_target_samples = [s for s in self.target_train_corpus if s.target_label != UNRELATED_TARGET and s.stance.name in permitted_stances]
        raw_stance_samples = [s for s in self.stance_train_corpus if s.target_label != UNRELATED_TARGET]
        target_samples = [self.encoder.encode(s, predict_task=PredictTask.TARGET, inference=False)
                          for s in tqdm(raw_target_samples, desc='Encoding target train corpus')]
        stance_samples = [self.encoder.encode(s, predict_task=PredictTask.STANCE, inference=False)
                          for s in tqdm(raw_stance_samples, desc='Encoding stance train corpus')]
        self.__n_stance = len(stance_samples)
        self.__train_ds = MapDataset(stance_samples + target_samples)
        self.__val_ds = MapDataset([self.encoder.encode(s, predict_task=PredictTask.STANCE, inference=True) for s in self.val_corpus])

    def train_dataloader(self):
        sampler = TaskSampler(np.arange(self.__n_stance), np.arange(self.__n_stance, len(self.__train_ds)), self.batch_size)
        return DataLoader(self.__train_ds, shuffle=False, batch_sampler=sampler, collate_fn=self.encoder.collate)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, shuffle=False, batch_size=self.batch_size, collate_fn=self.encoder.collate)
    

class SplitDataModule(BaseDataModule):
    def __init__(self,
                 corpora: List[StanceCorpus],
                 ratios: List[Tuple[float, float, float]],
                 batch_size: int = DEFAULT_BATCH_SIZE
                ):
        super().__init__()
        # Has to be set explicitly (see cli.py for an example)
        self.encoder: Encoder = None
        self.batch_size = batch_size
        self._corpora = corpora
        self._ratios = ratios
        assert len(self._corpora) == len(self._ratios)
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None
        for r in ratios:
            assert sum(r) == 1.


    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__test_ds:
            return

        train_dses = []
        val_dses = []
        test_dses = []
        for corpus, data_ratio in zip(self._corpora, self._ratios):
            samples = list(corpus)

            [train_raw, val_raw, test_raw] = random_split(MapDataset(samples), data_ratio)
            train_encode = lambda s: self.encoder.encode(s, inference=False)
            infer_encode = lambda s: self.encoder.encode(s, inference=True)
            train_ds = MapDataset(map(train_encode, tqdm(train_raw, desc=f"Encoding train samples for {corpus}")))
            val_ds = MapDataset(map(infer_encode, tqdm(val_raw, desc=f"Encoding val samples for {corpus}")))
            test_ds = MapDataset(map(infer_encode, tqdm(test_raw, desc=f"Encoding test samples for {corpus}")))

            train_dses.append(train_ds)
            val_dses.append(val_ds)
            test_dses.append(test_ds)
        self.__train_ds = ConcatDataset(train_dses)
        self.__val_ds = ConcatDataset(val_dses)
        self.__test_ds = ConcatDataset(test_dses)

    def train_dataloader(self):
        return DataLoader(self.__train_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.__val_ds,  batch_size=self.batch_size, collate_fn=self.encoder.collate)
    def test_dataloader(self):
        return DataLoader(self.__test_ds,  batch_size=self.batch_size, collate_fn=self.encoder.collate)