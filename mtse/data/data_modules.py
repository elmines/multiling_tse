# STL
from __future__ import annotations
import functools
import pdb
import pathlib
import csv
# 3rd Party
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Sampler
import lightning as L
from tqdm import tqdm
from typing import Tuple, List, Tuple, Optional
# Local
from .encoder import Encoder, PredictTask, keyed_scalar_stack
from .dataset import MapDataset
from .transforms import Transform
from .corpus import StanceCorpus
from .parse import DetCorpusType, CORPUS_PARSERS
from .target_pred import parse_target_preds
from ..constants import DEFAULT_BATCH_SIZE, UNRELATED_TARGET
from ..modules.mixins import TargetMixin

class BaseDataModule(L.LightningDataModule):
    """
    Dummy placeholder, just to constrain what classes the CLI permits
    """

    def __init__(self,
                 transforms: List[Transform] = []):
        super().__init__()
        self.transforms = transforms
        self._encoder: Encoder = None

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @encoder.setter
    def encoder(self, enc: Encoder):
        assert self._encoder is None
        self._encoder = enc
        for t in self.transforms:
            self._encoder.add_transform(t)

class TargetPredictionDataModule(BaseDataModule):
    """
    Only reads a CSV file of target predictions.
    Meant for use with the PassthroughModule
    """
    def __init__(self,
                 targets_path: pathlib.Path,
                 csv_paths: List[pathlib.Path],
                 with_generated: bool = False):
        super().__init__()
        # Inheriting from the TargetMixin breaks the super()
        # calls in L.LightningDataModule and its ancestors
        # Hence we use composition here instead
        target_mixin = TargetMixin(targets_path)
        self.targets = target_mixin.targets
        self.with_generated = with_generated

        self.csv_paths = csv_paths
        self.datasets = []

    def prepare_data(self):
        self.datasets.clear()
        for path in self.csv_paths:
            samples = []
            for pred in parse_target_preds(path):
                s = {
                    "target": torch.tensor(self.targets.index(pred.gt_target)),
                }
                if pred.mapped_target is not None:
                    s["target_preds"] = torch.tensor(self.targets.index(pred.mapped_target))
                if self.with_generated:
                    s['target_gens'] = pred.generated_targets
                    s["sample_inds"] = torch.full(pred.sample_id, len(pred.generated_targets))
                samples.append(s)
            self.datasets.append(MapDataset(samples))

    def _collate(self, samples):
        encoding = dict()
        encoding['target'] = keyed_scalar_stack(samples, 'target')
        if 'target_preds' in samples[0]:
            encoding['target_preds'] = keyed_scalar_stack(samples, 'target_preds')
        if 'target_gens' in samples[0]:
            encoding['target_gens'] = functools.reduce(lambda accum, el: accum + el, map(lambda x: x['target_gens'], samples))
            encoding['sample_inds'] = keyed_scalar_stack(samples, "sample_inds")
        return encoding
    
    def _dataloaders(self) -> List[torch.utils.data.DataLoader]:
        return [
            torch.utils.data.DataLoader(ds,
                                        batch_size=1024,
                                        collate_fn=self._collate) for ds in self.datasets]

    def predict_dataloader(self):
        return self._dataloaders()
    def test_dataloader(self):
        return self._dataloaders()

class PredictDataModule(BaseDataModule):
    def __init__(self,
                 corpora: List[StanceCorpus],
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **parent_kwargs
                 ):
        super().__init__(**parent_kwargs)
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
                 task_a_indices: np.ndarray,
                 task_b_indices: np.ndarray,
                 batch_size: int):
        self.task_a_indices = task_a_indices
        self.task_b_indices = task_b_indices
        self.batch_size = batch_size

        task_a_len = len(self.task_a_indices)
        self.__n_stance_batches = task_a_len // batch_size + bool(task_a_len % batch_size)
        task_b_len = len(self.task_b_indices)
        self.__n_target_batches = task_b_len // batch_size + bool(task_b_len % batch_size)

    def __len__(self):
        return self.__n_stance_batches + self.__n_target_batches

    def __iter__(self):
        permuted_stance_inds = np.random.permutation(self.task_a_indices)
        permuted_target_inds = np.random.permutation(self.task_b_indices)
        mixed_batches = []
        if self.__n_stance_batches:
            mixed_batches += np.array_split(permuted_stance_inds, self.__n_stance_batches)
        if self.__n_target_batches:
            mixed_batches += np.array_split(permuted_target_inds, self.__n_target_batches)
        random.shuffle(mixed_batches)
        mixed_batches = [torch.tensor(inds) for inds in mixed_batches]
        return iter(mixed_batches)

class MixedTrainingDataModule(BaseDataModule):
    def __init__(self,
                 stance_train_corpora: List[StanceCorpus],
                 stance_val_corpora: List[StanceCorpus],
                 keyword_corpus: Optional[StanceCorpus] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.keyword_corpus = keyword_corpus
        self.stance_train_corpora = stance_train_corpora
        self.stance_val_corpora = stance_val_corpora
        self.batch_size = batch_size

        self.__train_ds: Dataset = None
        self.__n_keyword: int = None
        self.__val_ds: Dataset = None

    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__n_keyword is not None:
            return

        train_samples = []
        if self.keyword_corpus is not None:
            keyword_samples = list(self.keyword_corpus)
            keyword_samples = [self.encoder.encode(s, inference=False) for s in tqdm(keyword_samples, desc='Encoding keyword samples')]
            self.__n_keyword = len(keyword_samples)
            train_samples += keyword_samples
        else:
            self.__n_keyword = 0

        train_stance_samples = [s for corp in self.stance_train_corpora for s in corp]
        train_stance_samples = [self.encoder.encode(s, inference=False) for s in tqdm(train_stance_samples, desc='Encoding train stance samples')]
        train_samples += train_stance_samples
        self.__train_ds = MapDataset(train_samples)

        val_stance_samples = [s for corp in self.stance_val_corpora for s in corp]
        self.__val_ds = MapDataset([self.encoder.encode(s, inference=False) for s in tqdm(val_stance_samples, desc='Encoding val stance samples')])

    def train_dataloader(self):
        sampler = TaskSampler(np.arange(self.__n_keyword), np.arange(self.__n_keyword, len(self.__train_ds)), self.batch_size)
        return DataLoader(self.__train_ds, shuffle=False, batch_sampler=sampler, collate_fn=self.encoder.collate)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, shuffle=False, batch_size=self.batch_size, collate_fn=self.encoder.collate)

class LiMultiTaskTrainingDataModule(BaseDataModule):
    """
    Datamodule for modelled from Li et al.'s approach of 
    training a BERT model to predict stance with an auxiliary
    target prediction objective
    """
    def __init__(self,
                 stance_train_corpus: StanceCorpus,
                 target_train_corpus: StanceCorpus,
                 val_corpus: StanceCorpus,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)

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
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **parent_kwargs
                ):
        super().__init__(**parent_kwargs)
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