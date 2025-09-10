#!/usr/bin/env python3
import pdb
import sys
import csv
import random
import os

data_dir = sys.argv[1]

in_path = os.path.join(data_dir, 'raw_et.csv')

label_map = {'1': 0, '2': 0, '3': 2, '4': 1, '5': 1, 'MH': 2}

favor_rows = []
against_rows = []
with open(in_path, 'r') as r:
    reader = csv.DictReader(r)
    for row in reader:
        s = row['Stance'] = label_map[row['stanceConsolidated']]
        if s == 1:
            favor_rows.append(row)
        elif s == 0:
            against_rows.append(row)

random.seed(0)
random.shuffle(favor_rows)
random.shuffle(against_rows)

favor_splindex = int(len(favor_rows) * 0.8)
against_splindex = int(len(against_rows) * 0.8)

train_rows = favor_rows[:favor_splindex] + against_rows[:against_splindex]
test_rows = favor_rows[favor_splindex:] + against_rows[against_splindex:]

out_entries = [
    (train_rows, "et_immigration_train.csv"),
    (test_rows, "et_immigration_test.csv")
]
for row_set, out_path in out_entries:
    with open(os.path.join(data_dir, out_path), 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Context", "Target", "Stance", "StanceType", "Lang"])
        writer.writeheader()
        for row in row_set:
            writer.writerow({
                "Context": row['sentence'],
                "Target": "Immigration",
                "Stance": row['Stance'],
                'StanceType': 'bi',
                'Lang': 'et'
            })