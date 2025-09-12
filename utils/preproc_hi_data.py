#!/usr/bin/env python3
import sys
import csv
import random
import os

in_dir = sys.argv[1]
out_dir = sys.argv[2]

def parse_file(path):
    tweet_id = None
    data = []
    with open(path, 'r') as r:
        for line in filter(lambda l: l, map(lambda l: l.strip(), r.readlines())):
            if tweet_id is None:
                tweet_id = line
            else:
                data.append((tweet_id, line))
                tweet_id = None
        assert tweet_id is None
        return data

text_entries = parse_file(os.path.join(in_dir, "hi_demonetisation_tweets.txt"))
stance_entries = parse_file(os.path.join(in_dir, "hi_demonetisation_stance.txt"))
assert len(text_entries) == len(stance_entries)

favor_rows = []
against_rows = []

label_map = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

for ((id_a, text), (id_b, stance)) in zip(text_entries, stance_entries):
    assert id_a == id_b
    stance = label_map[stance]
    if stance == 2:
        continue
    recipient = favor_rows if stance == 1 else against_rows
    recipient.append({
        "Context": text,
        "Target": "Demonetisation",
        "StanceType": 'bi',
        "Stance": stance,
        "Lang": "hi"
    })

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

def write_rows(out_path, rows):
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"])
        writer.writeheader()
        writer.writerows(rows)
write_rows(os.path.join(out_dir, "hi_demonetisation_train.csv"), train_rows)
write_rows(os.path.join(out_dir, "hi_demonetisation_val.csv"), val_rows)
write_rows(os.path.join(out_dir, "hi_demonetisation_test.csv"), test_rows)