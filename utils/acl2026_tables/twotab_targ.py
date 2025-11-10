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
    ("test/target/micro_f1", r"\Fmictarg"),
])

def head_matter():
    print(r"\begin{table}[]")
    print("% Table is programmatically generated")
    iprint(1, r"\centering")
    iprint(1, r"\begin{tabular}{lc}")
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

rowprint(r"Two-Pass TC \cite{li-etal-2023-new}", "75.59")
tc_entries = [
    ("MultiLiTClsWithBugWithScrub/seed*target_test/metrics.csv", "Two-Pass TC (Ours)"),
    ("MultiLiTClsWithScrub/seed*target_test/metrics.csv", f"{hspace}No Hashtag"),
    ("MultiLiTClsWithBug/seed*target_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiLiTCls/seed*target_test/metrics.csv", f"{hspace}No Hashtag + No Scrub"),

    ("MultiOneshotTClsWithBugWithScrub/seed*target_test/metrics.csv", "One-Pass TC"),
    ("MultiOneshotTClsWithScrub/seed*target_test/metrics.csv", f"{hspace}No Hashtag"),
    ("MultiOneshotTClsWithBug/seed*target_test/metrics.csv", f"{hspace}No Scrub"),
    ("MultiOneshotTCls/seed*target_test/metrics.csv", f"{hspace}No Hashtag + No Scrub"),
]
for patt, name in tc_entries:
    rowprint(name, *get_means(patt))
print_rule()
tg_entries = [
    ("MultiClassicTgen/seed*target_test_with_scrub/metrics.csv", "Two-Pass TG (Ours)"),
    ("MultiClassicTgen/seed*target_test/metrics.csv", f"{hspace}No Scrub"),

    ("MultiOneshotTgen/seed*target_test_with_scrub/metrics.csv", "One-Pass TG"),
    ("MultiOneshotTgen/seed*target_test/metrics.csv", f"{hspace}No Scrub"),
]
rowprint(r"Two-Pass TG \cite{li-etal-2023-new}", "48.31")
for patt, name in tg_entries:
    rowprint(name, *get_means(patt))

tail_matter("tab:targ_res", r"\Fmictarg\ for two-pass and one-pass algorithms.")
