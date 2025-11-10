#!/usr/bin/env python3
from common import read_test_row, round_percent
from collections import OrderedDict
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

metric_map = OrderedDict([
    # ("test/stance/bimacro_f1/semeval", r"SE"),
    # ("test/stance/bimacro_f1/am",      r"AM"),
    # ("test/stance/bimacro_f1/covid",   r"C19"),
    # ("test/stance/bimacro_f1/pstance", r"PS"),
    ("test/stance/avg_of_dataset_bimacro_f1", r"\Fmac"),
    ("test/stance/bimacro_f1", r"\Fmic"),
])

def head_matter():
    print(r"\begin{table}[]")
    print("% Table is programmatically generated")
    iprint(1, r"\centering")
    iprint(1, r"\begin{tabular}{lcc}")
    print_rule()
    rowprint("Model", *metric_map.values())
    print_rule()

def tail_matter(label, caption):
    iprint(1, r"\end{tabular}")
    iprint(1, r"\caption{" + caption + "}")
    iprint(1, r"\label{" + label + "}")
    print(r"\end{table}")


def get_means(pattern):
    paths = glob.glob(os.path.join(results_dir, pattern))
    assert len(paths) == 3, paths
    metric_samples = OrderedDict([(k, []) for k in metric_map])
    for p in paths:
        row = read_test_row(p)
        for m, v in metric_samples.items():
            v.append(float(row[m]))
    return tuple( round_percent(sum(v)/len(v)) for v in metric_samples.values() )

hspace = r"\hspace{5pt}+"

head_matter()

# li_met_vals = ["70.62", "64.85", "74.42", "81.67", "72.89", "73.01"]
li_met_vals = ["72.89", "73.01"]

rowprint(r"Two-Pass \cite{li-etal-2023-new}", *li_met_vals)
tc_entries = [
    ("MultiLiTClsWithBugWithScrub/seed*stance_test/metrics.csv", "Two-Pass (Ours)"),
    ("MultiLiTClsWithScrub/seed*stance_test/metrics.csv", f"{hspace}No Hashtag"),
    ("MultiLiTClsWithBug/seed*stance_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiLiTCls/seed*stance_test/metrics.csv", f"{hspace}No Hashtag + No Scrub"),

    # ("MultiClassicTgen/seed*stance_test_with_scrub/metrics.csv", "Two-Pass TG (Ours)"),
    # ("MultiClassicTgen/seed*stance_test/metrics.csv", f"{hspace}No Scrub"),

    ("MultiOneshotTClsWithBugWithScrub/seed*stance_test/metrics.csv", "One-Pass TC"),
    ("MultiOneshotTClsWithScrub/seed*stance_test/metrics.csv", f"{hspace}No Hashtag"),
    ("MultiOneshotTClsWithBug/seed*stance_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiOneshotTCls/seed*stance_test/metrics.csv", f"{hspace}No Hashtag + No Scrub"),
    ("MultiOneshotTgen/seed*stance_test_with_scrub/metrics.csv", "One-Pass TG"),
    ("MultiOneshotTgen/seed*stance_test/metrics.csv", f"{hspace}No Scrub"),
]
for patt, name in tc_entries:
    rowprint(name, *get_means(patt))
tail_matter("tab:stance_res", r"Stance \Favg\ scores one two-pass and one-pass models. For two-pass models, stance results are the same in both TC and TG scenarios.")
