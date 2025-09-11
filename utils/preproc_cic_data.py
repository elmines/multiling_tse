#!/usr/bin/env python3
import sys
import os
import csv

in_dir = sys.argv[1]
out_dir = sys.argv[2]

entries = []
entries.extend((f"ca_independence_{part}.csv", "ca") for part in ['train', 'val', 'test'])
entries.extend((f"es_independence_{part}.csv", "es") for part in ['train', 'val', 'test'])

stance_map = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

for (filename, lang) in entries:
    with open(os.path.join(in_dir, filename), 'r') as r:
        raw_rows = list(csv.DictReader(r, delimiter='\t'))
    clean_rows = [
        {
            "Context": row["TWEET"],
            "Target": "Catalonian Independence",
            "StanceType": "bi",
            "Stance": stance_map[row['LABEL']],
            "Lang": lang
        }
        for row in raw_rows if row['LABEL'] != 'NONE'
    ]
    with open(os.path.join(out_dir, filename), 'w') as w:
        writer = csv.DictWriter(w, fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"])
        writer.writeheader()
        writer.writerows(clean_rows)
    