#!/usr/bin/env python3
import sys
import csv
import random
import os

in_dir = sys.argv[1]
out_dir = sys.argv[2]

in_path = os.path.join(in_dir, 'et_immigration.csv')

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


train_rows = []
val_rows = []
test_rows = []
for class_data in [favor_rows, against_rows]:
    train_splindex = int(.7 * len(class_data))
    val_splindex = int(.8 * len(class_data))
    train_rows.extend(class_data[:train_splindex])
    val_rows.extend(class_data[train_splindex:val_splindex])
    test_rows.extend(class_data[val_splindex:])

out_entries = [
    (train_rows, "et_immigration_train.csv"),
    (val_rows, "et_immigration_val.csv"),
    (test_rows, "et_immigration_test.csv")
]
for row_set, out_path in out_entries:
    with open(os.path.join(out_dir, out_path), 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Context", "Target", "Stance", "StanceType", "Lang"], lineterminator='\n')
        writer.writeheader()
        for row in row_set:
            writer.writerow({
                "Context": row['sentence'],
                "Target": "Immigration",
                "Stance": row['Stance'],
                'StanceType': 'bi',
                'Lang': 'et'
            })