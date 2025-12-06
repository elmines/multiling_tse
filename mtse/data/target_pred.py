import dataclasses
from typing import Optional, Generator, Iterable, List
import csv
from ..constants import C_GT_TARGET, C_LANG, C_MAPPED_TARGET, C_SAMPLE, C_GENERATED_TARGET, C_UNTRANSLATED_TARGET

@dataclasses.dataclass
class TargetPred:
    sample_id: int
    gt_target: str
    generated_targets: List[str]
    untranslated_targets: List[str]
    mapped_target: Optional[str] = None
    lang: Optional[str] = None

def add_optional_field(kwargs_dict, row, csv_name, obj_name):
    if csv_name in row:
        value = row[csv_name].strip() or None
        if value:
            kwargs_dict[obj_name] = value

def parse_target_preds(in_path) -> Generator[TargetPred, None, None]:
    with open(in_path, 'r', encoding='utf-8') as r:
        reader = csv.DictReader(r, delimiter=',')
        last_sample_id = -1
        cur_pred: Optional[TargetPred] = None
        for row in reader:
            sample_id = int(row[C_SAMPLE])
            generated_target = row[C_GENERATED_TARGET]
            untranslated_target = row.get(C_UNTRANSLATED_TARGET, generated_target)
            if sample_id != last_sample_id:
                if cur_pred is not None:
                    yield cur_pred
                kwargs = {
                    "sample_id": sample_id,
                    "gt_target": row[C_GT_TARGET],
                    "generated_targets": [generated_target],
                    "untranslated_targets": [untranslated_target]
                }
                add_optional_field(kwargs, row, C_MAPPED_TARGET, 'mapped_target')
                add_optional_field(kwargs, row, C_LANG, 'lang')
                cur_pred = TargetPred(**kwargs)
                last_sample_id = sample_id
            else:
                cur_pred.generated_targets.append(generated_target)
                cur_pred.untranslated_targets.append(untranslated_target)
        if cur_pred is not None:
            yield cur_pred

class CsvTargetPredWriter:
    def __init__(self, writeable):
        self.__writer = csv.DictWriter(
            writeable,
            fieldnames=[C_SAMPLE, C_UNTRANSLATED_TARGET, C_GENERATED_TARGET, C_MAPPED_TARGET, C_GT_TARGET, C_LANG],
            lineterminator='\n'
        )
    
    def writeheader(self):
        self.__writer.writeheader()

    def writerows(self, preds: List[TargetPred]):
        for p in preds:
            mapped_target = p.mapped_target or ""
            for gtarg, utarg in zip(p.generated_targets, p.untranslated_targets):
                self.__writer.writerow({
                    C_SAMPLE: p.sample_id,
                    C_UNTRANSLATED_TARGET: utarg,
                    C_GENERATED_TARGET: gtarg,
                    C_MAPPED_TARGET: mapped_target,
                    C_GT_TARGET: p.gt_target,
                    C_LANG: p.lang
                })

def write_target_preds(out_path, preds: Iterable[TargetPred]):
    with open(out_path, 'w') as w:
        writer = CsvTargetPredWriter(w)
        writer.writeheader()
        writer.writerows(preds)

__all__ = ["TargetPred", "parse_target_preds", "write_target_preds"]