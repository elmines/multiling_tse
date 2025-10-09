#!/usr/bin/env python3
from collections import defaultdict
import re
import glob
import argparse
import os
import sys
import csv
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=pathlib.Path, required=True)
parser.add_argument("--task", choices=("tse", "tse_gt"), default='tse')
parser.add_argument("--noet", action="store_true")
parser.add_argument("-c", action="store_true", help="Columnwise")
parser.add_argument('-s', action='store_true', help='Skip header')
args = parser.parse_args()

in_dir = args.i
task = args.task
colwise = args.c
skipheader = args.s

langs = ["ca", "es", "et", "fr", "it", "zh"]


if task == "tse":
    paths = glob.glob(os.path.join(in_dir, "*tse_test", "metrics.csv"))
else:
    assert task == "tse_gt"
    paths = glob.glob(os.path.join(in_dir, "*tse_test_gt", "metrics.csv"))
assert len(paths) == 5

fold_scores = defaultdict(list)

noet = args.noet or "Noet" in str(args.i)

for p in paths:
    with open(p, 'r') as r:
        row = next(csv.DictReader(r))

    macro_samples = []

    for lang in langs:
        if lang == 'et' and noet:
            continue
        tp_patt = re.compile(f"^test/tse/tp/{lang}")
        fp_patt = re.compile(f"^test/tse/fp_wrong(targ|stance)/{lang}")
        fn_patt = re.compile(f"^test/tse/fn_wrong(targ|stance)/{lang}")

        tps = 0
        for k in filter(tp_patt.search, row):
            tps += float(row[k])
        fps = 0
        for k in filter(fp_patt.search, row):
            fps += float(row[k])
        fns = 0
        for k in filter(fn_patt.search, row):
            fns += float(row[k])
        prec = tps / (tps + fps + 1e-6)
        rec = tps / (tps + fns + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)

        fold_scores[lang].append(f1)
        macro_samples.append(f1)
    fold_scores['macro'].append( sum(macro_samples) / len(macro_samples) )

avgs = {k:sum(v)/len(v) for k,v in fold_scores.items()}
if "et" not in avgs:
    avgs['et'] = '-'

fieldnames = ["macro", *langs]
if colwise:
    if skipheader:
        rows = [ [avgs[field]] for field in fieldnames ]
    else:
        rows = [ [field, avgs[field]]  for field in fieldnames]
    writer = csv.writer(sys.stdout)
    writer.writerows(rows)
else:
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    if not skipheader:
        writer.writeheader()
    writer.writerow(avgs)
