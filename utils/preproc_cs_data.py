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

def readlines(infile):
    with open(os.path.join(in_dir, infile), 'r') as r:
        return list(filter(lambda l: l, map(lambda l: l.strip(), r.readlines())))

smoking_texts = readlines('smoking-gold.txt')
smoking_labels = readlines('smoking-gold.labels')
zeman_texts = readlines('zeman.txt')
zeman_labels = readlines('zeman.labels')

label_map = {
    "PROTI": 0,
    "PRO": 1,
}

entries = [
    (smoking_texts, smoking_labels, "Smoking Ban in Restaurants"),
    (zeman_texts, zeman_labels, "Miloš Zeman"),
]

train_rows = []
val_rows = []
test_rows = []

random.seed(0)
for (text_list, label_list, target) in entries:
    favor_samples = []
    against_samples = []
    for (text, label) in filter(lambda tl: tl[1] != 'NIC', zip(text_list, label_list)):
        stance = label_map[label]
        sample = {
            "Context": text,
            "Target": target,
            "StanceType": "bi",
            "Stance": stance,
            "Lang": "cs"
        }
        if stance == 0:
            against_samples.append(sample)
        else:
            favor_samples.append(sample)
    random.shuffle(favor_samples)
    random.shuffle(against_samples)
    for class_data in [favor_samples, against_samples]:
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
write_rows(os.path.join(out_dir, "cs_train.csv"), train_rows)
write_rows(os.path.join(out_dir, "cs_val.csv"), val_rows)
write_rows(os.path.join(out_dir, "cs_test.csv"), test_rows)