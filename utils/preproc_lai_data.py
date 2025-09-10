#!/usr/bin/env python3
import pdb
import sys
import csv
import random
import os

data_dir = sys.argv[1]
to_convert = [
    ("referendum_it.csv", "lai_referendum_{split}.csv", "2016 Italian Constitution Referendum", "it"),
    ("lepen_fr.csv", "lai_lepen_{split}.csv", "Marine LePen", "fr"),
    ("macron_fr.csv", "lai_macron_{split}.csv", "Emmanuel Macron", "fr"),
]

label_map = {
    'favor': 1,
    'FAVOUR': 1,
    'agains': 0,
    'AGAINST': 0,
    'none': 2,
    'NONE': 2
}
random.seed(0)


for (in_name, out_template, target, lang) in to_convert:
    with open(os.path.join(data_dir, in_name), 'r') as r:
        raw_rows = list(csv.DictReader(r))
    favor_samples = []
    against_samples = []
    for row in raw_rows:
        s = row['Stance'] = label_map[row['Stance']]
        if s == 1:
            favor_samples.append(row)
        elif s == 0:
            against_samples.append(row)
        random.shuffle(favor_samples)
        random.shuffle(against_samples)

        favor_splindex = int(.8 * len(favor_samples))
        against_splindex = int(.8 * len(against_samples))

        def write_rows(out_path, rows):
            with open(out_path, 'w') as w:
                writer = csv.DictWriter(w, fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "Context": row['Tweet'],
                        "Target": target,
                        "Stance": row['Stance'],
                        "StanceType": "bi",
                        "Lang": lang
                    })

        write_rows(
            os.path.join(data_dir, out_template.format(split='train')),
            favor_samples[:favor_splindex] + against_samples[:against_splindex]) 
        write_rows(
            os.path.join(data_dir, out_template.format(split='test')),
            favor_samples[favor_splindex:] + against_samples[against_splindex:]
        )