#!/usr/bin/env python3
from common import read_test_row, round_percent
import glob
import os
import sys

results_dir = sys.argv[1]

def iprint(indent: int, *args, **kwargs):
    print("\t" * indent, end="")
    print(*args, **kwargs)

def rowprint(*args, **kwargs):
    iprint(2, *args, sep=' & ', end=" \\\\\n")
def print_rule():
    iprint(2, r"\hline")

def head_matter():
    print(r"\begin{table}[]")
    print("% Table is programmatically generated")
    iprint(1, r"\centering")
    iprint(1, r"\begin{tabular}{lcc}")
    print_rule()
    rowprint("Model", "Predicted", "GT")
    print_rule()

def tail_matter(label, caption):
    iprint(1, r"\end{tabular}")
    iprint(1, r"\caption{" + caption + "}")
    iprint(1, r"\label{" + label + "}")
    print(r"\end{table}")

def get_means(pattern):
    pattern = os.path.join(results_dir, pattern)
    tse_patt = pattern
    tse_paths = glob.glob(tse_patt)
    tse_vals = [float(read_test_row(p)["test/tse/f1"]) for p in tse_paths]
    gt_tse_patt = pattern.replace("tse_test", "tse_test_gt")
    gt_tse_vals = [float(read_test_row(p)["test/tse/f1"]) for p in glob.glob(gt_tse_patt)]
    assert len(tse_vals) == 3, tse_paths
    assert len(gt_tse_vals) == 3
    return round_percent(sum(tse_vals)/len(tse_vals)) , round_percent(sum(gt_tse_vals)/len(tse_vals))

hspace = r"\hspace{5pt}+"

head_matter()
rowprint(r"Two-Pass TC \cite{li-etal-2023-new}", "53.30", "75.28")
tc_entries = [
    # Target classification
    ("MultiLiTClsWithBugWithScrub/seed*tse_test/metrics.csv", "Two-Pass TC (Ours)"),
    ("MultiLiTClsWithScrub/seed*tse_test/metrics.csv", f"{hspace}No Hashtag"),
    ("MultiLiTClsWithBug/seed*tse_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiLiTCls/seed*tse_test/metrics.csv", f"{hspace}No Hashtag + No Scrub"),
    ("MultiOneshotTClsWithBugWithScrub/seed*tse_test/metrics.csv", "One-Pass TC"),
    ("MultiOneshotTClsWithScrub/seed*tse_test/metrics.csv", f"{hspace}No Hashtag"),
    ("MultiOneshotTClsWithBug/seed*tse_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiOneshotTCls/seed*tse_test/metrics.csv", f"{hspace}No Hashtag + No Scrub")
]
for patt, name in tc_entries:
    rowprint(name, *get_means(patt))
print_rule()
## Target Generation
rowprint(r"Two-Pass TG \cite{li-etal-2023-new}", "38.92", "79.49")
tg_entries = [
    ("MultiClassicTgen/seed*tse_test_with_scrub/metrics.csv", "Two-Pass TG (Ours)"),
    ("MultiClassicTgen/seed*tse_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiOneshotTgen/seed*tse_test_with_scrub/metrics.csv", "One-Pass TG"),
    ("MultiOneshotTgen/seed*tse_test/metrics.csv", f"{hspace}No Scrub"),
]
for patt, name in tg_entries:
    rowprint(name, *get_means(patt))
tail_matter("tab:tse_res", r"TSE F1 Scores, using both Predicted and GT targets.")