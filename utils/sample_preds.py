#!/usr/bin/env python3
"""
Sample multilingual predictions made.
The primary purpose of this is for our accompanying paper.
"""
from collections import defaultdict
import sys
import os
import random
import glob
import csv

in_dir = sys.argv[1]
random.seed(0)

fold_paths = glob.glob(os.path.join(in_dir, f"fold*_seed*_full_target_preds.csv")) 
# Randomly choose a fold
fold_path = random.choice(fold_paths)
print(fold_path)

lang_sets = [
    ["ca_catalonia"],
    ["es_catalonia"],
    ["et_immigration"],
    ["fr_macron", "fr_lepen"],
    ["it_sardinia"],
    ["zh_firecracker", "zh_iphone", "zh_russia", "zh_shenzhen", "zh_twochild"]
]

part_dict = defaultdict(list)
with open(fold_path, 'r') as r:
    for row in csv.DictReader(r):
        part_dict[row['Partition']].append(row)
part_dict = dict(part_dict)


def row_print(*args, **kwargs):
    print("\t", end='')
    print(*args, sep=' & ', end=' \\\\\n', **kwargs)
def print_rule():
    print('\t', end='')
    print(r"\hline")

print(r"\begin{table*}")
print(r"\small")
print(r"\begin{center}")
print(r"\begin{tabularx}{\linewidth}{|X|X|X|X|}")
print_rule()
row_print("Document", "Generated Targets", "Mapped Target", "Groundtruth Target")
print_rule()

for lang_set in lang_sets:
    # Randomly choose a file
    partition = random.choice(lang_set)
    rows = part_dict[partition]
    row = random.choice(rows)

    document = row['Context']
    document = document.replace('\n', ' ').replace('%', '\\%').replace('#', '\\#')
    stance = row['GoldStance']
    chosen = row['ChosenCandidate']

    candidates = row['Candidates'][1:-1].split(';')
    candidates_strbuilder = []
    for c in candidates:
        if c == chosen:
            candidates_strbuilder.append("\\textbf{" + c + "}")
        else:
            candidates_strbuilder.append(c)
    candidates_str = ';'.join(candidates_strbuilder)

    row_print(document, candidates_str, row['MappedTarget'], row['GoldTarget'])
    print_rule()

print(r"\end{tabularx}")
print(r"\end{center}")
print(r"\caption{Randomly selected target predictions. The chosen candidate is bolded.}")
print(r"\label{tab:sample_preds}")
print(r"\end{table*}")