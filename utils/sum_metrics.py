#!/usr/bin/env python3
import glob
import argparse
import os
import sys
import csv
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=pathlib.Path, required=True)
parser.add_argument("--task", choices=("target", "tse", "stance", "tse_gt"), default='target')
args = parser.parse_args()

in_dir = args.i
task = args.task

all_labels = [
    "ca_catalonia",
    "ca_unrelated",
    "es_catalonia",
    "es_unrelated",
    "et_immigration",
    "et_unrelated",
    "fr_lepen",
    "fr_macron",
    "fr_unrelated",
    "it_sardinia",
    "it_unrelated",
    "zh_firecracker",
    "zh_iphone",
    "zh_russia",
    "zh_shenzhen",
    "zh_twochild",
    "zh_unrelated"
]

if task == "target":
    labels = all_labels
    paths = glob.glob(os.path.join(in_dir, "*target_test", "metrics.csv"))
    metric_keys = ["test/target/micro_f1"] + [f"test/target/micro_f1/{l}" for l in labels]
elif task == 'stance':
    labels = [l for l in all_labels if "unrelated" not in l]
    paths = glob.glob(os.path.join(in_dir, "*stance_test", "metrics.csv"))
    metric_keys = ["test/stance/bimacro_f1"] + [f"test/stance/bimacro_f1/{l}" for l in labels]
elif task == 'tse_gt':
    labels = [l for l in all_labels if "unrelated" not in l]
    paths = glob.glob(os.path.join(in_dir, "*tse_test_gt", "metrics.csv"))
    metric_keys = ["test/tse/f1"] + [f"test/tse/f1/{l}" for l in labels]
else:
    assert task == 'tse'
    labels = [l for l in all_labels if "unrelated" not in l]
    paths = glob.glob(os.path.join(in_dir, "*tse_test", "metrics.csv"))
    metric_keys = ["test/tse/f1"] + [f"test/tse/f1/{l}" for l in labels]
metric_entries = {k:[] for k in metric_keys}
assert len(paths) == 5

for p in paths:
    with open(p, 'r') as r:
        row = next(csv.DictReader(r))
    for k in metric_keys:
        metric_entries[k].append(float(row.get(k, 0.0)))

metric_means = {k:sum(v)/len(v) for k,v in metric_entries.items()}
writer = csv.DictWriter(sys.stdout, fieldnames=metric_keys)
writer.writeheader()
writer.writerow(metric_means)
