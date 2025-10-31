# STL
from __future__ import annotations
import os
import pathlib
import glob
import pdb
from collections import Counter, defaultdict
from itertools import batched, chain
# 3rd Party
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Sampler
import lightning as L
from tqdm import tqdm
from typing import Tuple, List, Tuple, Optional, Generator
# Local
from .encoder import Encoder, PredictTask, keyed_scalar_stack, concat_lists
from .target_pred import TargetPred
from .dataset import MapDataset
from .transforms import Transform, TargetRename
from .corpus import StanceCorpus, TargetInputType, CorpusLike
from .parse import DetCorpusType, CORPUS_PARSERS, parse_standard
from .target_pred import parse_target_preds
from .sample import Sample
from ..constants import DEFAULT_BATCH_SIZE, UNRELATED_TARGET, LANG_TO_ID, INDEPENDENCE_TARGETS, INDEPENDENCE
from ..modules.mixins import TargetMixin

class BaseDataModule(L.LightningDataModule):
    """
    Dummy placeholder, just to constrain what classes the CLI permits
    """

    def __init__(self,
                 transforms: Optional[List[Transform]] = None):
        super().__init__()
        self.transforms = transforms or []
        self._encoder: Encoder = None

    @property
    def testloader_labels(self) -> Optional[List[str]]:
        return None

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @encoder.setter
    def encoder(self, enc: Encoder):
        assert self._encoder is None
        self._encoder = enc
        for t in self.transforms:
            self._encoder.add_transform(t)


def get_dataloader_labels(corpora: List[CorpusLike]):
    labels_list = [c.name if c.name else f"loader_{i}"
                   for i, c in enumerate(corpora)]
    label_counts = Counter(labels_list)
    dups = [k for k,v in label_counts.items() if v > 1]
    if dups:
        raise ValueError(f"Duplicate corpora labels {dups}")
    return sorted(label_counts)

class PathSampler(Sampler):
    """
    Guarantees that a batch will only have samples with the same source path
    """
    def __init__(self, source_paths: List[str], batch_size: int):
        inds_by_path = defaultdict(list)
        for ind, p in sorted(enumerate(source_paths), key=lambda pair: pair[1]):
            inds_by_path[p].append(ind)
        self.__inds_by_path = inds_by_path
        self.__batch_size = batch_size

    def __iter__(self):
        return chain(*[batched(inds, self.__batch_size) for p, inds in sorted(self.__inds_by_path.items())])
    
    def __len__(self):
        lens = [len(v) for v in self.__inds_by_path.values()]
        return sum( (v // self.__batch_size) + bool(v % self.__batch_size) for v in lens)

 
class PattDataModule(BaseDataModule):
    """
    Looks for files with suffixies _train.csv, _val.csv, _test.csv
    """
    def __init__(self,
                 train_corpus: Optional[CorpusLike] = None,
                 val_corpus: Optional[CorpusLike] = None,
                 test_corpora: Optional[List[CorpusLike]] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 transforms: Optional[List[Transform]] = None):
        super().__init__(transforms=transforms)
        self.batch_size = batch_size
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None

        self.__train_corpus = StanceCorpus.make_corpus(train_corpus) if train_corpus else None
        self.__val_corpus = StanceCorpus.make_corpus(val_corpus) if val_corpus else None
        self.__test_corpora = [StanceCorpus.make_corpus(c) for c in (test_corpora or [])]
        self.__testloader_labels = get_dataloader_labels(self.__test_corpora)

        self.__testloader_labels = [c.name for c in self.__test_corpora]
        # Allow multiple datasets for eval purposes
        self.__test_datasets: List[Dataset] = None


    @property
    def testloader_labels(self):
        return self.__testloader_labels

    def _parse_path(self, path) -> Generator[Sample, None, None]:
        target_preds_path = None
        if self.preds_dir is not None:
            label = PattDataModule._extract_label(path)
            target_preds_path = os.path.join(self.preds_dir, label + ".target_preds.csv")
            if not os.path.exists(target_preds_path):
                raise ValueError(f'Could not find target preds for "{label}" at expected path "{target_preds_path}"')

        yield from StanceCorpus(path,
                                corpus_type="standard",
                                target_input=self.target_input,
                                target_preds_path=target_preds_path)

    def _setup_train(self):
        if self.__train_ds is not None:
            return
        if self.__train_corpus:
            self.__train_ds = MapDataset(map(lambda s: self.encoder.encode(s, inference=False), self.__train_corpus))
        if self.__val_corpus:
            self.__val_ds = MapDataset(map(lambda s: self.encoder.encode(s, inference=True), self.__val_corpus))

    def _setup_test(self):
        if self.__test_datasets is not None:
            return
        self.__test_datasets = [
            MapDataset( map(lambda s: self.encoder.encode(s, inference=True), c) )
            for c in self.__test_corpora
        ]

    def setup(self, stage):
        if stage == 'fit':
            self._setup_train()
        else:
            self._setup_test()

    def train_dataloader(self):
        return DataLoader(self.__train_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.__val_ds,  batch_size=self.batch_size, collate_fn=self.encoder.collate)

    def test_dataloader(self):
        loaders = []
        for ds in self.__test_datasets:
            sampler = PathSampler(
                [s['source_path'] for s in ds],
                batch_size=self.batch_size
            )
            loaders.append(torch.utils.data.DataLoader(ds,
                                    batch_sampler=sampler,
                                    collate_fn=self.encoder.collate)
            )
        return loaders
    def predict_dataloader(self):
        return self.test_dataloader()

class TargetPredictionDataModule(BaseDataModule):
    """
    Only reads a CSV file of target predictions.
    Meant for use with the PassthroughModule
    """


    def __init__(self,
                 data_dir: pathlib.Path,
                 targets_path: pathlib.Path,
                 suffix_pattern: str = ".target_gens.csv",
                 exclude_patterns: List[str] = [],
                 with_generated: bool = False,
                 with_untranslated: bool = False,
                 transforms: List[Transform] = []
                 ):
        super().__init__()
        # Inheriting from the TargetMixin breaks the super()
        # calls in L.LightningDataModule and its ancestors
        # Hence we use composition here instead
        self.data_dir = data_dir
        target_mixin = TargetMixin(targets_path)
        self.targets = target_mixin.targets
        self.with_generated = with_generated
        self.with_untranslated = with_untranslated

        self.datasets = []
        self.transforms = transforms

        all_paths = glob.glob(os.path.join(self.data_dir, f"*{suffix_pattern}"))
        excluded = []
        for patt in exclude_patterns:
            excluded.extend(glob.glob(os.path.join(self.data_dir, patt)))
        self.__test_paths = sorted(set(all_paths) - set(excluded))
        self.__testloader_labels = [os.path.basename(p).split(suffix_pattern)[0] for p in self.__test_paths]

    @property
    def testloader_labels(self):
        return self.__testloader_labels

    def prepare_data(self):
        self.datasets.clear()
        for path in self.__test_paths:
            samples = []
            pred_iter = parse_target_preds(path)

            for pred in pred_iter:
                for t in self.transforms:
                    t(pred) # Transforms are in-place

                s = {
                    "target": torch.tensor(self.targets.index(pred.gt_target)),
                }
                s['lang'] = torch.tensor(LANG_TO_ID[pred.lang], dtype=torch.long)
                if pred.mapped_target is not None:
                    s["target_preds"] = torch.tensor(self.targets.index(pred.mapped_target))
                if self.with_generated or self.with_untranslated:
                    s["sample_inds"] = torch.full((len(pred.generated_targets),), pred.sample_id)
                    if self.with_generated:
                        s['target_gens'] = pred.generated_targets
                    if self.with_untranslated:
                        s['target_untrans'] = pred.untranslated_targets
                samples.append(s)
            self.datasets.append(MapDataset(samples))

    def _collate(self, samples):
        encoding = dict()
        encoding['target'] = keyed_scalar_stack(samples, 'target')
        encoding['lang'] = keyed_scalar_stack(samples, 'lang')
        if 'target_preds' in samples[0]:
            encoding['target_preds'] = keyed_scalar_stack(samples, 'target_preds')
        for k in ['target_gens', 'target_untrans']:
            if k in samples[0]:
                encoding[k] = concat_lists(samples, k)
        if 'sample_inds' in samples[0]:
            encoding['sample_inds'] = torch.concatenate([s['sample_inds'] for s in samples])
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

class OneshotTgenDataModule(BaseDataModule):
    def __init__(self,
                 keyword_train_corpus: CorpusLike,
                 stance_train_corpus: CorpusLike,
                 stance_val_corpus: CorpusLike,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **parent_kwargs,
                 ):
        super().__init__(**parent_kwargs)
        self.keyword_train_corpus = StanceCorpus.make_corpus(keyword_train_corpus)
        self.stance_train_corpus = StanceCorpus.make_corpus(stance_train_corpus)
        self.stance_val_corpus = StanceCorpus.make_corpus(stance_val_corpus)
        self.batch_size = batch_size

        self.__train_ds: Dataset = None
        self.__n_keyword: int = None
        self.__val_ds: Dataset = None

    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__n_keyword is not None:
            return

        train_samples = []
        keyword_samples = list(self.keyword_train_corpus)
        keyword_samples = [self.encoder.encode(s, inference=False) for s in tqdm(keyword_samples, desc='Encoding keyword samples')]
        self.__n_keyword = len(keyword_samples)
        train_samples += keyword_samples

        train_stance_samples = list(self.stance_train_corpus)
        train_stance_samples = [self.encoder.encode(s, inference=False) for s in tqdm(train_stance_samples, desc='Encoding train stance samples')]
        train_samples += train_stance_samples
        self.__train_ds = MapDataset(train_samples)

        val_stance_samples = list(self.stance_val_corpus)
        self.__val_ds = MapDataset([self.encoder.encode(s, inference=False) for s in tqdm(val_stance_samples, desc='Encoding val stance samples')])

    def train_dataloader(self):
        sampler = TaskSampler(np.arange(self.__n_keyword), np.arange(self.__n_keyword, len(self.__train_ds)), self.batch_size)
        return DataLoader(self.__train_ds, shuffle=False, batch_sampler=sampler, collate_fn=self.encoder.collate)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, shuffle=False, batch_size=self.batch_size, collate_fn=self.encoder.collate)

class ClassicMultiTaskTrainingDataModule(BaseDataModule):
    """
    Datamodule for modelled from Li et al.'s approach of 
    training a BERT model to predict stance with an auxiliary
    target prediction objective
    """
    def __init__(self,
                 target_train_corpus: CorpusLike,
                 stance_train_corpus: CorpusLike,
                 val_corpus: CorpusLike,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.__target_train_corpus = StanceCorpus.make_corpus(target_train_corpus)
        self.__stance_train_corpus = StanceCorpus.make_corpus(stance_train_corpus)
        self.__val_corpus = StanceCorpus.make_corpus(val_corpus)
        self.batch_size = batch_size

        self.__train_ds: Dataset = None
        self.__n_stance: int = None
        self.__val_ds: Dataset = None

    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__n_stance is not None:
            return

        
        # For the auxiliary target task, Li et al. use the unrelated samples,
        # and samples with targets and a non-Neutral stance
        permitted_stances = {'favor', 'against'}
        raw_target_samples = [s for s in self.__target_train_corpus if s.target_label != UNRELATED_TARGET and s.stance.name in permitted_stances]
        raw_stance_samples = [s for s in self.__stance_train_corpus if s.target_label != UNRELATED_TARGET]
        target_samples = [self.encoder.encode(s, predict_task=PredictTask.TARGET, inference=False)
                          for s in tqdm(raw_target_samples, desc='Encoding target train corpus')]
        stance_samples = [self.encoder.encode(s, predict_task=PredictTask.STANCE, inference=False)
                          for s in tqdm(raw_stance_samples, desc='Encoding stance train corpus')]
        self.__n_stance = len(stance_samples)
        self.__train_ds = MapDataset(stance_samples + target_samples)
        self.__val_ds = MapDataset([self.encoder.encode(s, predict_task=PredictTask.STANCE, inference=True) for s in self.__val_corpus])

    def train_dataloader(self):
        sampler = TaskSampler(np.arange(self.__n_stance), np.arange(self.__n_stance, len(self.__train_ds)), self.batch_size)
        return DataLoader(self.__train_ds, shuffle=False, batch_sampler=sampler, collate_fn=self.encoder.collate)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, shuffle=False, batch_size=self.batch_size, collate_fn=self.encoder.collate)
    
