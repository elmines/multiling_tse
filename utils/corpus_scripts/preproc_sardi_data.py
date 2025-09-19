#!/usr/bin/env python3
import sys
import csv
import random
import os

TARGET = "Sardinian Independence"
FIELDNAMES = [
    "Context",
    "Target",
    "Stance",
    "StanceType",
    "Lang"
]

in_dir = sys.argv[1]
out_dir = sys.argv[2]

train_in_path       = os.path.join(in_dir, "it_sardinia_train.csv")
test_in_path        = os.path.join(in_dir, "it_sardinia_test.csv")
test_labels_in_path = os.path.join(in_dir, "it_sardinia_test_labels.csv")

label_map = {'AGAINST': 0, 'FAVOR': 1}


with open(test_in_path, 'r') as r:
    raw_test_rows = list(csv.DictReader(r))
with open(test_labels_in_path, 'r') as r:
    raw_test_label_rows = list(csv.DictReader(r))
assert len(raw_test_rows) == len(raw_test_label_rows)
test_rows = []
for (data, label) in zip(raw_test_rows, raw_test_label_rows):
    assert data['tweet_id'] == label['tweet_id']
    raw_label = label['label']    
    if raw_label not in label_map:
        continue
    test_rows.append({
        "Context": data['text'],
        "Target": TARGET,
        "Stance": label_map[raw_label],
        "StanceType": "bi",
        "Lang": "it"
    })

def write_rows(p, rows):
    with open(p, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=FIELDNAMES, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

write_rows(os.path.join(out_dir, "it_sardinia_test.csv"), test_rows)

with open(train_in_path, 'r') as r:
    raw_rows = list(csv.DictReader(r))
against_rows = []
favor_rows = []
for row in filter(lambda row: row['label'] in label_map, raw_rows):
    label = label_map[row['label']]
    cleaned = {
        "Context": row['text'],
        'Target': TARGET,
        'Stance': label,
        'StanceType': 'bi',
        'Lang': 'it'
    }
    if label == 0:
        against_rows.append(cleaned)
    else:
        favor_rows.append(cleaned)
random.shuffle(against_rows)
random.shuffle(favor_rows)

against_splindex = int(.875 * len(against_rows))
favor_splindex = int(.875 * len(favor_rows))
write_rows(os.path.join(out_dir, "it_sardinia_train.csv"), against_rows[:against_splindex] + favor_rows[:favor_splindex])
write_rows(os.path.join(out_dir, "it_sardinia_val.csv"), against_rows[against_splindex:] + favor_rows[favor_splindex:])
