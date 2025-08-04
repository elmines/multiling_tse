import pathlib
from typing import Optional, List
import functools
# 3rd Party
from tqdm import tqdm
# Local
from .parse import DetCorpusType, CORPUS_PARSERS
from .transforms import Transform

class StanceCorpus:
    def __init__(self,
                 corpus_type: DetCorpusType,
                 path: pathlib.Path,
                 target_preds_path: Optional[pathlib.Path] = None,
                 transforms: List[Transform] = []):
        self._parse_fn = CORPUS_PARSERS[corpus_type]
        self._path = path
        self._target_preds_path = target_preds_path

        # Combine those transforms into one function
        self._transform = lambda s: functools.reduce(lambda accum, t: t(accum), transforms, s)

    def __str__(self):
        return f"<StanceCorpus path='{self._path}'>"

    @staticmethod
    def _iter_targets(target_path):
        with open(target_path, 'r') as r:
            for line in r:
                yield line.strip()

    def __iter__(self):
        sample_iter = self._parse_fn(self._path)
        if self._target_preds_path is not None:
            target_iter = StanceCorpus._iter_targets(self._target_preds_path)
            def combined_iter():
                for sample, target in zip(sample_iter, target_iter):
                    sample.target_prediction = target
                    yield sample
            raw_iter = combined_iter()
            desc = f"Parsing {self._path} and {self._target_preds_path}"
        else:
            raw_iter = sample_iter
            desc = f"Parsing {self._path}"
        trans_iter = map(self._transform, raw_iter)
        return iter(tqdm(trans_iter, desc=desc))