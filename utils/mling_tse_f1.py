#!/usr/bin/env python3
from collections import defaultdict
import re
import argparse
import sys
import csv
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=pathlib.Path, nargs="+", required=True)
parser.add_argument("-c", action="store_true", help="Columnwise")
parser.add_argument('-s', action='store_true', help='Skip header')
args = parser.parse_args()

paths = args.i
colwise = args.c
skipheader = args.s

langs = ["ca", "es", "et", "fr", "it", "zh"]
print(len(paths))

fold_scores = defaultdict(list)

def compute_f1(tps, fps, fns):
    prec = tps / (tps + fps + 1e-6)
    rec = tps / (tps + fns + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return f1

for p in paths:
    with open(p, 'r') as r:
        row = next(csv.DictReader(r))

    macro_samples = []
    tp_all = 0
    fp_all = 0
    fn_all = 0

    for lang in langs:
        tp_patt = re.compile(f"^test/tse/tp/{lang}")
        fp_patt = re.compile(f"^test/tse/fp_wrong(targ|stance)/{lang}")
        fn_patt = re.compile(f"^test/tse/fn_wrong(targ|stance)/{lang}")

        tps = 0
        for k in filter(tp_patt.search, row):
            tps += float(row[k])
        tp_all += tps
        fps = 0
        for k in filter(fp_patt.search, row):
            fps += float(row[k])
        fp_all += fps
        fns = 0
        for k in filter(fn_patt.search, row):
            fns += float(row[k])
        fn_all += fns
        f1 = compute_f1(tps, fps, fns)
        fold_scores[lang].append(f1)
        macro_samples.append(f1)
    fold_scores['macro'].append( sum(macro_samples) / len(macro_samples) )
    fold_scores['micro'].append( compute_f1(tp_all, fp_all, fn_all) )

avgs = {k:sum(v)/len(v) for k,v in fold_scores.items()}

fieldnames = [*langs, "micro", "macro"]
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
