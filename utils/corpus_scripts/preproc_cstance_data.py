#!/usr/bin/env python3
import sys
import csv
import random
import os
from itertools import islice

in_dir = sys.argv[1]
out_dir = sys.argv[2]

in_path = os.path.join(in_dir, 'zh_cstance.csv')
MAX_ROWS = 1000

with open(in_path, 'r', encoding='utf-8-sig') as r:
    raw_rows = list(islice(csv.DictReader(r), MAX_ROWS))
rows = [{"Context": raw_row['Text'],
        "Target": "Unrelated",
        'StanceType': 'tri',
        "Stance": 2, # Neutral
        'Lang': 'zh'} for raw_row in raw_rows]
random.seed(0)
random.shuffle(rows)

train_splindex = int(.7 * len(rows))
val_splindex = int(.8 * len(rows))

def write_rows(out_path, rows):
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
write_rows(os.path.join(out_dir, "zh_unrelated_train.csv"), rows[:train_splindex])
write_rows(os.path.join(out_dir, "zh_unrelated_val.csv"), rows[train_splindex:val_splindex])
write_rows(os.path.join(out_dir, "zh_unrelated_test.csv"), rows[val_splindex:])
