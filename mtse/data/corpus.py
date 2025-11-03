from __future__ import annotations
import os
import glob
import pathlib
import csv
from typing import Optional, List, Literal, Iterable
import copy
from itertools import islice, chain
# 3rd Party
from tqdm import tqdm
# Local
from .sample import Sample
from .parse import DetCorpusType, CORPUS_PARSERS
from .transforms import Transform

TargetInputType = Literal['pred', 'label']

def get_paths(patterns: List[pathlib.Path]):
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(str(p)))
    return sorted(set(all_paths))

class StanceCorpus(Iterable[Sample]):

    def __init__(self,
                 patts: List[pathlib.Path],
                 corpus_type: DetCorpusType = 'standard',
                 transforms: List[Transform] = [],
                 limit_n: Optional[int] = None,
                 name: Optional[str] = None):
        self._paths = get_paths(patts)
        assert self._paths, f"Found no paths from {patts}"
        self._parse_fn = CORPUS_PARSERS[corpus_type]
        self._transforms = transforms
        self._limit_n = limit_n

        if name is None:
            self.name = StanceCorpus._extract_label(self._paths[0])
        else:
            self.name = name

    @staticmethod
    def _extract_label(file_path):
        return os.path.basename(file_path).split(".")[0]

    @staticmethod
    def make_corpus(corp_like: CorpusLike):
        if isinstance(corp_like, StanceCorpus):
            return corp_like
        elif isinstance(corp_like, list):
            return StanceCorpus(corp_like)
        else:
            return StanceCorpus([corp_like])

    def _apply_transforms(self, sample: Sample):
        if self._transforms:
            # Only waste resources making a copy if we have transforms to apply
            # This is why transforms are in-place: we don't have to make a copy for each transform
            s = copy.deepcopy(sample)
            for t in self._transforms:
                t(s)
        else:
            s = sample
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
        corp_iterables = []
        for p in self._paths:
            trans_iter = map(self._apply_transforms, self._parse_fn(p))
            if self._limit_n is not None:
                trans_iter = islice(trans_iter, self._limit_n)
            corp_iterables.append(tqdm(trans_iter, desc=f"Parsing {p}"))
        return chain(*corp_iterables)

CorpusLike = StanceCorpus | pathlib.Path | List[pathlib.Path]