import dataclasses
from typing import Optional, Generator, Iterable, List
import csv

@dataclasses.dataclass
class TargetPred:
    sample_id: int
    gt_target: str
    generated_targets: List[str]
    mapped_target: Optional[str] = None

def parse_target_preds(in_path) -> Generator[TargetPred, None, None]:
    with open(in_path, 'r', encoding='utf-8') as r:
        reader = csv.DictReader(r, delimiter=',')
        last_sample_id = -1
        cur_pred: Optional[TargetPred] = None
        for row in reader:
            sample_id = int(row['Sample'])
            if sample_id != last_sample_id:
                if cur_pred is not None:
                    yield cur_pred
                kwargs = {
                    "sample_id": sample_id,
                    "gt_target": row["GT Target"],
                    "generated_targets": [row['Generated Target']]
                }
                if "Mapped Target" in row:
                    mapped_target = row['Mapped Target'].strip() or None
                    if mapped_target:
                        kwargs['mapped_target'] = mapped_target
                cur_pred = TargetPred(**kwargs)
            else:
                cur_pred.generated_targets(row['Generated Target'])
        if cur_pred is not None:
            yield cur_pred

def write_target_preds(out_path, preds: Iterable[TargetPred]):
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Sample", "Generated Target", "Mapped Target", "GT Target"])
        writer.writeheader()
        for p in preds:
            mapped_target = p.mapped_target or ""
            for gtarg in p.generated_targets:
                writer.writerow({
                    "Sample": p.sample_id,
                    "Generated Target": gtarg,
                    "Mapped Target": mapped_target,
                    "GT Target": p.gt_target
                })

__all__ = ["TargetPred", "parse_target_preds", "write_target_preds"]