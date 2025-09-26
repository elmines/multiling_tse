import dataclasses
from typing import Optional, Generator, Iterable, List
import csv

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
            sample_id = int(row['Sample'])
            generated_target = row['Generated Target']
            untranslated_target = row.get('Untranslated Target', generated_target)
            if sample_id != last_sample_id:
                if cur_pred is not None:
                    yield cur_pred
                kwargs = {
                    "sample_id": sample_id,
                    "gt_target": row["GT Target"],
                    "generated_targets": [generated_target],
                    "untranslated_targets": [untranslated_target]
                }
                add_optional_field(kwargs, row, 'Mapped Target', 'mapped_target')
                add_optional_field(kwargs, row, 'Lang', 'lang')
                cur_pred = TargetPred(**kwargs)
                last_sample_id = sample_id
            else:
                cur_pred.generated_targets.append(generated_target)
                cur_pred.untranslated_targets.append(untranslated_target)
        if cur_pred is not None:
            yield cur_pred

def write_target_preds(out_path, preds: Iterable[TargetPred]):
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Sample", "Untranslated Target", "Generated Target", "Mapped Target", "GT Target", "Lang"])
        writer.writeheader()
        for p in preds:
            mapped_target = p.mapped_target or ""
            for gtarg, utarg in zip(p.generated_targets, p.untranslated_targets):
                writer.writerow({
                    "Sample": p.sample_id,
                    "Untranslated Target": utarg,
                    "Generated Target": gtarg,
                    "Mapped Target": mapped_target,
                    "GT Target": p.gt_target,
                    "Lang": p.lang
                })

__all__ = ["TargetPred", "parse_target_preds", "write_target_preds"]