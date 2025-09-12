#!/usr/bin/env python3
import pdb
import sys
import csv
import random
import os

in_dir = sys.argv[1]
out_dir = sys.argv[2]
to_convert = [
    ("it_ref.csv", "it_ref_{split}.csv", "2016 Italian Constitution Referendum", "it"),
    ("fr_lepen.csv", "fr_lepen_{split}.csv", "Marine LePen", "fr"),
    ("fr_macron.csv", "fr_macron_{split}.csv", "Emmanuel Macron", "fr"),
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
    with open(os.path.join(in_dir, in_name), 'r') as r:
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

        train_rows = []
        val_rows = []
        test_rows = []
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
                for row in rows:
                    writer.writerow({
                        "Context": row['Tweet'],
                        "Target": target,
                        "Stance": row['Stance'],
                        "StanceType": "bi",
                        "Lang": lang
                    })

        write_rows(
            os.path.join(out_dir, out_template.format(split='train')),
            train_rows
        )
        write_rows(
            os.path.join(out_dir, out_template.format(split='val')),
            val_rows
        )
        write_rows(
            os.path.join(out_dir, out_template.format(split='test')),
            test_rows
        )