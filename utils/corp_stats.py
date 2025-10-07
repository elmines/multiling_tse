#!/usr/bin/env python3

import csv
import sys
import glob
import os
from collections import defaultdict

fold_dir = sys.argv[1]

counts = defaultdict(lambda: [0, 0, 0])

for p in glob.glob(os.path.join(fold_dir, "*.csv")):
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

targ_len = 51
print("\\hline")
table_print("Language", "Target", "\\# Against", "\\# Favor", "\\# Neutral")
print("\\hline")
for (lang, targ), [against_count, fav_count, neut_count] in counts:
    if len(targ) > targ_len:
        targ = f"{targ[:targ_len]}..."
    table_print(lang, targ, against_count, fav_count, neut_count)
print("\\hline")

