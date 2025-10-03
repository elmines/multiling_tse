#!/usr/bin/env python3

import argparse
import os
import sys
import csv
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=pathlib.Path, required=True)
parser.add_argument("--merged", action="store_true")
parser.add_argument("--task", choices=("target", "tse", "stance"), default='target')
args = parser.parse_args()

in_dir = args.i
merged = bool(args.merged)
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

merged_stem = "_merged" if merged else ""
path_template = "fold{fold}_seed0" + merged_stem 
if task == "target":
    labels = all_labels
    path_template += "_target_test"
    metric_keys = ["test/target/micro_f1"] + [f"test/target/micro_f1/{l}" for l in labels]
elif task == 'stance':
    labels = [l for l in all_labels if "unrelated" not in l]
    path_template += "_stance_test"
    metric_keys = ["test/stance/bimacro_f1"] + [f"test/stance/bimacro_f1/{l}" for l in labels]
else:
    assert task == 'tse'
    labels = [l for l in all_labels if "unrelated" not in l]
    path_template += "_tse_test"
    metric_keys = ["test/tse/f1"] + [f"test/tse/f1/{l}" for l in labels]
metric_entries = {k:[] for k in metric_keys}

for i in range(5):
    p = os.path.join(in_dir, path_template.format(fold=i), "metrics.csv")
    with open(p, 'r') as r:
        row = next(csv.DictReader(r))
    for k in metric_keys:
        metric_entries[k].append(float(row[k]))

metric_means = {k:sum(v)/len(v) for k,v in metric_entries.items()}
writer = csv.DictWriter(sys.stdout, fieldnames=metric_keys)
writer.writeheader()
writer.writerow(metric_means)