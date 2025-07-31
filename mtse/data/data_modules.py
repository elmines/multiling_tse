# STL
from __future__ import annotations
import pathlib
# 3rd Party
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Sampler
import lightning as L
from tqdm import tqdm
from typing import Tuple, List, Tuple, Sequence
# Local
from .encoder import Encoder
from .dataset import MapDataset
from .corpus import StanceCorpus
from .parse import DetCorpusType, CORPUS_PARSERS
from ..constants import DEFAULT_BATCH_SIZE

class BaseDataModule(L.LightningDataModule):
    """
    Dummy placeholder, just to constrain what classes the CLI permits
    """

class PredictDataModule(BaseDataModule):
    def __init__(self, corpus: StanceCorpus, batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.encoder: Encoder = None
        self.batch_size = batch_size
        self.corpus = corpus
        self.__ds: Dataset = None

    def setup(self, stage):
        if self.__ds is not None:
            return
        corpus = self.corpus
        samples = list(tqdm(corpus.parse_fn(corpus.path), desc=f"Parsing {corpus.path}"))
        self.__ds = MapDataset(map(lambda s: self.encoder.encode(s, inference=True), samples))

    def predict_dataloader(self):
        return DataLoader(self.__ds, batch_size=self.batch_size, collate_fn=self.encoder.collate)
    def test_dataloader(self):
        return DataLoader(self.__ds, batch_size=self.batch_size, collate_fn=self.encoder.collate)
    

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
            samples = list(tqdm(corpus.parse_fn(corpus.path), desc=f"Parsing {corpus.path}"))

            [train_raw, val_raw, test_raw] = random_split(MapDataset(samples), data_ratio)
            train_encode = lambda s: self.encoder.encode(s, inference=False)
            infer_encode = lambda s: self.encoder.encode(s, inference=True)
            train_ds = MapDataset(map(train_encode, tqdm(train_raw, desc=f"Encoding train samples for {corpus.path}")))
            val_ds = MapDataset(map(infer_encode, tqdm(val_raw, desc=f"Encoding val samples for {corpus.path}")))
            test_ds = MapDataset(map(infer_encode, tqdm(test_raw, desc=f"Encoding test samples for {corpus.path}")))

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