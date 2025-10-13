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

fold_paths = sorted(glob.glob(os.path.join(in_dir, f"fold*_seed*_full_target_preds.csv")))
if len(sys.argv) > 2:
    fold_ind = int(sys.argv[2])
    fold_path = fold_paths[fold_ind]
else:
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
print(r"\scriptsize{")
print(r"\begin{tabular}{p{.09in}p{2.06in}p{1.045in}", end="")
print(r">{\raggedright\arraybackslash}p{1.09in}", end="")
print(r">{\raggedright\arraybackslash}p{1.08in}", end="")
print(r"}")
print_rule()
print(r"\hspace*{-.03in}")

row_print(r"\textbf{Lang}",
          r"\textbf{Document}",
          r"\textbf{Generated Targets}",
          r"\textbf{Mapped Target}",
          r"\textbf{Groundtruth Target}")
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

    lang = partition.split('_')[0]
    row_print(lang, document, candidates_str, row['MappedTarget'], row['GoldTarget'])
    print_rule()

print(r"\end{tabular}")
print(r"}")
print(r"\caption{Randomly selected target predictions. The chosen candidate is \textbf{bold}.}")
print(r"\label{tab:sample_preds}")
print(r"\end{table*}")