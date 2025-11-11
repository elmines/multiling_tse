#!/usr/bin/python3
import sys
import csv
import os
import random
import re
# Local

pred_dir = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2

entries = [
    ("SE", "semeval"),
    ("AM", "am"),
    ("C19", "covid"),
    ("PS", "pstance"),
    ("Unrelated", "unrelated")
]

stance_map = {"0": "AGAINST", "1": "FAVOR", "2": "NEUTRAL"}

WHITESPACE = re.compile('\s+')
def latex_clean(s):
    s = s.replace('&', r'\&').replace('#', r'\#')
    return WHITESPACE.sub(' ', s)

random.seed(seed)
for table_label, file_label in entries:
    predictions_path = os.path.join(pred_dir, f"{file_label}.test.csv.full.csv")
    with open(predictions_path, 'r') as r:
        sample = random.choice(list(csv.DictReader(r)))
    assert sample['StanceType'] == 'tri'

    print(table_label,
        latex_clean(sample["Context"]),
        latex_clean(sample['GeneratedTarget']),
        latex_clean(sample['MappedTarget']),
        latex_clean(sample['Target']),
        stance_map[sample['StancePred']],
        stance_map[sample['Stance']],
        sep=' & ',
        end=' \\\\\n'
    )