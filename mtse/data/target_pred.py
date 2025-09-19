import dataclasses
from typing import Optional, Generator, Iterable
import csv

@dataclasses.dataclass
class TargetPred:
    generated_target: str
    gt_target: str
    mapped_target: Optional[str] = None

def parse_target_preds(in_path) -> Generator[TargetPred, None, None]:
    def f(row):
        return TargetPred(
            generated_target=row['Generated Target'],
            mapped_target=row['Mapped Target'].strip() or None,
            gt_target=row['GT Target']
        )
    with open(in_path, 'r', encoding='utf-8') as r:
        yield from map(f, csv.DictReader(r, delimiter=','))

def write_target_preds(out_path, preds: Iterable[TargetPred]):
    def f(pred: TargetPred):
        return {
            "Generated Target": pred.generated_target,
            "Mapped Target": pred.mapped_target or "",
            "GT Target": pred.gt_target
        }
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Generated Target", "Mapped Target", "GT Target"])
        writer.writeheader()
        writer.writerows(map(f, preds))

__all__ = ["TargetPred", "parse_target_preds", "write_target_preds"]