import pathlib
import csv
from typing import Optional, List, Literal
import functools
import copy
from itertools import islice
# 3rd Party
from tqdm import tqdm
# Local
from .sample import Sample
from .parse import DetCorpusType, CORPUS_PARSERS
from .transforms import Transform

TargetInputType = Literal['pred', 'label']

class StanceCorpus:

    class SetTargetInput(Transform):
        def __init__(self, target_input: TargetInputType):
            self.target_input = target_input
        def __call__(self, sample: Sample):
            assert sample.target_input is None
            if self.target_input == 'pred':
                sample.target_input = sample.target_pred
            elif self.target_input == 'label':
                sample.target_input = sample.target_label
            else:
                raise ValueError(f"Invalid target_input = {self.target_input}")

    def __init__(self,
                 corpus_type: DetCorpusType,
                 path: pathlib.Path,
                 target_preds_path: Optional[pathlib.Path] = None,
                 transforms: List[Transform] = [],
                 target_input: TargetInputType = 'label',
                 limit_n: Optional[int] = None):
        self._parse_fn = CORPUS_PARSERS[corpus_type]
        self._path = path
        self._target_preds_path = target_preds_path
        self._transforms = [StanceCorpus.SetTargetInput(target_input)] + transforms
        self._limit_n = limit_n

        # Combine those transforms into one function
        self._transform = lambda s: functools.reduce(lambda accum, t: t(accum), transforms, s)

    def _apply_transforms(self, sample: Sample):
        # This is why transforms are in-place: we don't have to make a copy for each transform
        s = copy.deepcopy(sample)
        for t in self._transforms:
            t(s)
        return s

    def __str__(self):
        return f"<StanceCorpus path='{self._path}'>"

    @staticmethod
    def _iter_targets(target_path):
        with open(target_path, 'r') as r:
            reader = csv.DictReader(r)
            for row in reader:
                yield row['Mapped Target']

    def __iter__(self):
        sample_iter = self._parse_fn(self._path)
        if self._target_preds_path is not None:
            target_iter = StanceCorpus._iter_targets(self._target_preds_path)
            def combined_iter():
                for sample, target in zip(sample_iter, target_iter):
                    sample.target_pred = target
                    yield sample
            raw_iter = combined_iter()
            desc = f"Parsing {self._path} and {self._target_preds_path}"
        else:
            raw_iter = sample_iter
            desc = f"Parsing {self._path}"
        trans_iter = map(self._apply_transforms, raw_iter)
        if self._limit_n is not None:
            trans_iter = islice(trans_iter, self._limit_n)
        return iter(tqdm(trans_iter, desc=desc))