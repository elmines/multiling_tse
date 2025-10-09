#!/usr/bin/env python3

import csv
import sys
import glob
import os
from collections import defaultdict
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=pathlib.Path, required=True)
parser.add_argument('--skip_unrelated', action='store_true')

parser.add_argument('--disp', choices=('latex', 'lang_tot'), default='latex')

args = parser.parse_args()

fold_dir = args.i
skip_unrel = args.skip_unrelated
disp = args.disp

counts = defaultdict(lambda: [0, 0, 0])

for p in glob.glob(os.path.join(fold_dir, "*.csv")):
    if skip_unrel and 'unrelated' in p:
        continue
    with open(p, 'r') as r:
        rows = list(csv.DictReader(r))
    for row in rows:
        lang = row['Lang']
        targ = row['Target']
        stance = int(row['Stance'])
        counts[lang, targ][stance] += 1

counts = dict(counts)
lang_map = {
    "ca": "Catalan",
    "es": "Spanish",
    "et": "Estonian",
    "it": "Italian",
    "zh": "Mandarin",
    "fr": "French"
}
counts = { (lang_map[l], targ) : count for (l, targ),count in counts.items()}
counts = sorted(counts.items(), key=lambda keyval: keyval[0])

def table_print(*args, **kwargs):
    print(*args, sep=' & ', end=' \\\\\n', **kwargs)
    pass

if disp == 'latex':
    targ_len = 51
    print("\\hline")
    table_print("Language", "Target", "\\# Against", "\\# Favor", "\\# Neutral")
    print("\\hline")
    for (lang, targ), [against_count, fav_count, neut_count] in counts:
        if len(targ) > targ_len:
            targ = f"{targ[:targ_len]}..."
        table_print(lang, targ, against_count, fav_count, neut_count)
    print("\\hline")
else:
    assert disp == 'lang_tot'
    lang_tots = {l:0 for l in lang_map.values()}
    for (lang, _), count_arr in counts:
        lang_tots[lang] += sum(count_arr)
    for lang, count in sorted(lang_tots.items(), key=lambda kv: kv[0]):
        print(lang, count)
