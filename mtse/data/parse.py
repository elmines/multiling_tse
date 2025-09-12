from typing import Generator, Dict, Literal, Callable, Iterable
import json
import sys
import os
import csv

from .stance import TriStance, STANCE_TYPE_MAP, get_stance_type_str
from .sample import Sample, SampleType
from ..constants import TARGET_DELIMITER

def write_standard(out_path, samples: Iterable[Sample]):
    def f(s: Sample):
        return {
            "Context": s.context,
            "Target": s.target_label,
            "StanceType": get_stance_type_str(type(s.stance)),
            "Stance": int(s.stance),
            "Lang": s.lang if s.lang else ""
        }

    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=[
            "Context",
            "Target",
            "StanceType",
            "Stance",
            "Lang"
        ],
        lineterminator='\n')
        writer.writeheader()
        writer.writerows(map(f, samples))
        pass

def parse_standard(corpus_path) -> Generator[Sample, None, None]:
    def f(row):
        stance_type = STANCE_TYPE_MAP[row['StanceType']]
        stance_val = stance_type(int(row['Stance']))
        s = Sample(
            context=row['Context'],
            target_label=row['Target'],
            stance=stance_val,
            lang=row['Lang']
        )
        return s
    with open(corpus_path, 'r', encoding='utf-8') as r:
        yield from map(f, csv.DictReader(r, delimiter=','))

def parse_yingjie(corpus_path) -> Generator[Sample, None, None]:
    str2strance = {
        "FAVOR": TriStance.favor,
        "NONE": TriStance.neutral,
        "AGAINST": TriStance.against,
        "Dummy Stance": TriStance.neutral
    }
    def f(row):
        return Sample(context=row['Tweet'],
                      target_label=row['Target'],
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
                      target_label=None, # Using this as the "unrelated" target dataset
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
                target_label=row['TARGET'],
                stance=stance_map[stance],
                lang='zh'
            )
    if discarded:
        print(f"Discarded {discarded} samples from {corpus_path}", file=sys.stderr)

def parse_kptimes(corpus_path: os.PathLike):
    with open(corpus_path, 'r', encoding='utf-8') as r:
        for line in r:
            json_doc = json.loads(line)
            context = json_doc['abstract']
            target_phrase = TARGET_DELIMITER.join(json_doc['keyphrases'])
            yield Sample(
                context=context,
                target_label=target_phrase,
                stance=TriStance.neutral,
                lang='en',
                sample_type=SampleType.KG
            )

DetCorpusType = Literal['standard', 'nlpcc', 'cstance', 'li', 'kptimes']

StanceParser = Callable[[os.PathLike], Generator[Sample, None, None]]
"""
Function taking a file path and returning a generator of samples
"""

CORPUS_PARSERS: Dict[DetCorpusType, StanceParser] = {
    "standard": parse_standard,
    "nlpcc": parse_nlpcc,
    "cstance": parse_cstance,
    "li": parse_yingjie,
    "kptimes": parse_kptimes
}