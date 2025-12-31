#!/usr/bin/env python3
import glob
import argparse
import os
import sys
import csv
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-i", nargs="+", type=pathlib.Path, required=True)
args = parser.parse_args()

paths = args.i

metric_keys = [f"test/target/f1/class_{i}" for i in range(11)] + ["test/target/micro_f1"] + [f"test/target/macro_f1"]
metric_entries = {k:[] for k in metric_keys}
assert len(paths) == 15

for p in paths:
    with open(p, 'r') as r:
        row = next(csv.DictReader(r))
    for k in metric_keys:
        metric_entries[k].append(float(row.get(k, 0.0)))

metric_means = {k:sum(v)/len(v) for k,v in metric_entries.items()}
writer = csv.DictWriter(sys.stdout, fieldnames=metric_keys)
writer.writeheader()
writer.writerow(metric_means)
