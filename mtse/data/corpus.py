import pathlib
from typing import List
import functools

from .parse import DetCorpusType, CORPUS_PARSERS

class StanceCorpus:
    def __init__(self,
                 path: pathlib.Path,
                 corpus_type: DetCorpusType):
        self._parse_fn = CORPUS_PARSERS[corpus_type]
        self.path = path
        self.transforms = []

    def parse_fn(self, *args, **kwargs):
        for sample in self._parse_fn(*args, **kwargs):
            sample = functools.reduce(lambda s, f: f(s), self.transforms, sample)
            yield sample
