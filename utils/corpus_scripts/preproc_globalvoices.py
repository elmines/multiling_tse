#!/usr/bin/env python3

import os
import csv
import random
import sys

in_dir = sys.argv[1]
# in_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "..", "data", "multiling", "raw")

out_dir = sys.argv[2]
# out_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "..", "data", "multiling")

entries = [
    ("ca", 3300),
    ("es", 3300),
    ("fr", 500),
    ("it", 2000),
]

MIN_CHARS = 128

def write_texts(lang, part, texts):
    out_path = os.path.join(out_dir, f"{lang}_unrelated_{part}.csv")
    rows = [
        {"Context": t, "Target": "Unrelated", "StanceType": "tri", "Stance": 3, "Lang": lang}
        for t in texts
    ]
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w,
                                fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

for (lang, n_samples) in entries:
    in_path = os.path.join(in_dir, f"{lang}_globalvoices.txt")
    with open(in_path, 'r') as r:
        samples = [l.strip() for l in r.readlines()]
    samples = list(filter(lambda l: len(l) >= MIN_CHARS, samples))
    assert len(samples) >= n_samples
    random.seed(0)
    random.shuffle(samples)
    samples = samples[:n_samples]

    train_splindex = int(.7 * len(samples))
    val_splindex = int(.8 * len(samples))

    write_texts(lang, "train", samples[:train_splindex])
    write_texts(lang, "val", samples[train_splindex:val_splindex])
    write_texts(lang, "test", samples[val_splindex:])
