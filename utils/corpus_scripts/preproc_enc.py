#!/usr/bin/env python3
"""
Preprocess the Estonian National Corpus
"""

import csv
import random
import json
import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

in_path = os.path.join(in_dir, "et_unrelated.jsonl")
samples = []
n_samples = 700
MIN_CHARS = 128
with open(in_path, 'r') as r:
    for line in r.readlines():
        try:
            json_doc = json.loads(line)
        except json.decoder.JSONDecodeError:
            continue
        samples.append(json_doc['text'])

samples = [t for t in samples if len(t) >= MIN_CHARS]
assert len(samples) >= n_samples
random.seed(0)
random.shuffle(samples)
samples = samples[:n_samples]

train_splindex = int(.7 * len(samples))
val_splindex = int(.8 * len(samples))

def write_samples(part, texts):
    out_path = os.path.join(out_dir, f"et_unrelated_{part}.csv")
    rows = [
        {"Context": t, "Target": "Unrelated", "StanceType": "tri", "Stance": 3, "Lang": 'et'}
        for t in texts
    ]
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w,
                                fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

write_samples("train", samples[:train_splindex])
write_samples("val", samples[train_splindex:val_splindex])
write_samples("test", samples[val_splindex:])


