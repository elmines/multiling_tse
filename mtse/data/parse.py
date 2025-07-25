from typing import Generator, Dict, Literal, Callable
import sys
import os
import csv

from .stance import TriStance
from .sample import Sample

def parse_nlpcc(corpus_path: os.PathLike):
    stance_map = {"NONE": TriStance.neutral, "AGAINST": TriStance.against, "FAVOR": TriStance.favor}
    discarded = 0
    with open(corpus_path, 'r') as r:
        reader = csv.DictReader(r, delimiter='\t')
        for row in reader:
            stance = row['STANCE']
            if not stance:
                discarded += 1
                continue
            yield Sample(
                context=row['TEXT'],
                target=row['TARGET'],
                stance=stance_map[stance]
            )
    if discarded:
        print(f"Discarded {discarded} samples from {corpus_path}", file=sys.stderr)

DetCorpusType = Literal['nlpcc']

StanceParser = Callable[[os.PathLike], Generator[Sample, None, None]]
"""
Function taking a file path and returning a generator of samples
"""

CORPUS_PARSERS: Dict[DetCorpusType, StanceParser] = {
    "nlpcc": parse_nlpcc
}