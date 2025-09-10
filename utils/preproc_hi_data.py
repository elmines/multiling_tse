#!/usr/bin/env python3
import sys
import csv
import random
import os

data_dir = sys.argv[1]

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

text_entries = parse_file(os.path.join(data_dir, "Notebandi_tweets.txt"))
stance_entries = parse_file(os.path.join(data_dir, "Notebandi_tweets_stance.txt"))
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
favor_splindex = int(.8 * len(favor_rows))
against_splindex = int(.8 * len(against_rows))

def write_rows(out_path, rows):
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"])
        writer.writeheader()
        writer.writerows(rows)
write_rows(os.path.join(data_dir, "demonetisation_train.csv"), favor_rows[:favor_splindex] + against_rows[:against_splindex] )
write_rows(os.path.join(data_dir, "demonetisation_test.csv"), favor_rows[favor_splindex:] + against_rows[against_splindex:] )