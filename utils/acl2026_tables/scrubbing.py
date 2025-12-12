#!/usr/bin/python3
import sys
import csv
import os
import re
# Local

pred_dir = sys.argv[1]

TARGET_WORDS =  [
    'Joe',
    'Biden',
    'Bernie',
    'Sanders',
    'Donald',
    'Trump',
    'abortion',
    'cloning',
    'death',
    'penalty',
    'gun',
    'control',
    'marijuana',
    'legalization',
    'minimum',
    'wage',
    'nuclear',
    'energy',
    'school',
    'uniforms',
    'Atheism',
    'Feminist',
    'Movement',
    'Hillary',
    'Clinton',
    'face',
    'masks',
    'fauci',
    'stay',
    'home',
    'school',
    'closures',
    'orders'
]

TARGET_PATT = re.compile('|'.join(TARGET_WORDS), flags=re.IGNORECASE)

entries = [
    ("SE", "semeval"),
    ("AM", "am"),
    ("C19", "covid"),
    ("PS", "pstance"),
    ("Unrelated", "unrelated")
]

stance_map = {"0": "AGAINST", "1": "FAVOR", "2": "NEUTRAL"}

WHITESPACE = re.compile('\s+')
LATEX_CHAR = re.compile('|'.join(['&', r'\#', r'\$', '_']))
def latex_clean(s):
    s = LATEX_CHAR.sub(lambda m: f"\\{m.group(0)}", s, count=0)
    # s = s.replace('&', r'\&').replace('#', r'\#').replace('$', r'\$')
    return WHITESPACE.sub(' ', s)

for table_label, file_label in entries:
    predictions_path = os.path.join(pred_dir, f"{file_label}.test.csv.full.csv")

    with open(predictions_path, 'r') as r:
        samples = list(csv.DictReader(r))

    repl_counts = []
    for s in samples:
        assert s['StanceType'] == 'tri'
        s['Context'], c = TARGET_PATT.subn(lambda m: r"\st{" + m.group(0) + "}", s['Context'], count=0)
        repl_counts.append(c)

    sorted_inds = sorted(range(len(repl_counts)), key=lambda i: repl_counts[i], reverse=True)
    most_replaced = sorted_inds[0]
    sample = samples[most_replaced]


    print(table_label,
        latex_clean(sample['Context']),
        latex_clean(sample['GeneratedTarget']),
        latex_clean(sample['Target']),
        sep=' & ',
        end=' \\\\\n'
    )
