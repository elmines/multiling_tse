from typing import Generator, Dict, Literal, Callable
import sys
import os
import csv

from .stance import TriStance
from .sample import Sample

def parse_yingjie(corpus_path) -> Generator[Sample, None, None]:
    str2strance = {
        "FAVOR": TriStance.favor,
        "NONE": TriStance.neutral,
        "AGAINST": TriStance.against,
        "Dummy Stance": TriStance.neutral
    }
    def f(row):
        target = row['Target'] if row['Target'] != 'Unrelated' else None
        return Sample(context=row['Tweet'],
                      target=target,
                      stance=str2strance[row['Stance']],
                      lang='en')
    with open(corpus_path, 'r', encoding='ISO-8859-1') as r:
        yield from map(f, csv.DictReader(r, delimiter=','))

def parse_cstance(corpus_path) -> Generator[Sample, None, None]:
    strstance2 = {
        "支持": TriStance.favor,
        "反对": TriStance.against,
        "中立": TriStance.neutral
    }
    def f(row):
        return Sample(context=row['Text'],
                      target=None, # Using this as the "unrelated" target dataset
                      stance=strstance2[row['Stance 1']],
                      lang='zh')
    with open(corpus_path, 'r', encoding='utf-8-sig') as r:
        yield from map(f, csv.DictReader(r, delimiter=','))

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
                stance=stance_map[stance],
                lang='zh'
            )
    if discarded:
        print(f"Discarded {discarded} samples from {corpus_path}", file=sys.stderr)

DetCorpusType = Literal['nlpcc', 'cstance', 'li']

StanceParser = Callable[[os.PathLike], Generator[Sample, None, None]]
"""
Function taking a file path and returning a generator of samples
"""

CORPUS_PARSERS: Dict[DetCorpusType, StanceParser] = {
    "nlpcc": parse_nlpcc,
    "cstance": parse_cstance,
    "li": parse_yingjie
}