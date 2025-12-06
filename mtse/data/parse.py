from typing import Generator, Dict, Literal, Callable, Iterable
import json
import os
import csv

from .stance import TriStance, STANCE_TYPE_MAP, get_stance_type_str
from .sample import Sample, SampleType
from ..constants import TARGET_DELIMITER, C_GT_STANCE, C_GT_TARGET, C_LANG

def write_standard(out_path, samples: Iterable[Sample]):
    def f(s: Sample):
        return {
            "Context": s.context,
            C_GT_TARGET: s.target_label,
            "StanceType": get_stance_type_str(type(s.stance)),
            C_GT_STANCE: int(s.stance),
            "Lang": s.lang if s.lang else ""
        }

    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=[
            "Context",
            C_GT_TARGET,
            "StanceType",
            C_GT_STANCE,
            "Lang"
        ],
        lineterminator='\n')
        writer.writeheader()
        writer.writerows(map(f, samples))

def parse_standard(corpus_path) -> Generator[Sample, None, None]:
    def f(row):
        stance_type = STANCE_TYPE_MAP[row['StanceType']]
        stance_val = stance_type(int(row[C_GT_STANCE]))
        s = Sample(
            context=row['Context'],
            target_label=row[C_GT_TARGET],
            stance=stance_val,
            lang=row[C_LANG],
            source_path=corpus_path
        )
        return s
    with open(corpus_path, 'r', encoding='utf-8') as r:
        yield from map(f, csv.DictReader(r, delimiter=','))

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
                sample_type=SampleType.KG,
                source_path=corpus_path
            )

DetCorpusType = Literal['standard', 'kptimes']

StanceParser = Callable[[os.PathLike], Generator[Sample, None, None]]
"""
Function taking a file path and returning a generator of samples
"""

# TODO: Just convert the KPTimes data to standard format so we don't have to
# have distinct parsers anymore?
CORPUS_PARSERS: Dict[DetCorpusType, StanceParser] = {
    "standard": parse_standard,
    "kptimes": parse_kptimes
}