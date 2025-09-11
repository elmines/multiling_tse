#!/usr/bin/env python3
import pdb
import random
import sys
import csv
import random
import os

in_dir = sys.argv[1]
out_dir = sys.argv[2]

target_map = {
    "IphoneSE": "IphoneSE",
    "春节放鞭炮": "Setting off firecrackers during the Spring Festival",
    "俄罗斯在叙利亚的反恐行动": "Russia's counter-terrorism operations in Syria",
    "开放二胎": "Allowing second births",
    "深圳禁摩限电": "Shenzhen bans motorcycles and imposes electricity restrictions",
}
label_map = {
    'FAVOR': 1,
    'AGAINST': 0,
}
targets = sorted(target_map.values())
samples_by_target = {t:[] for t in targets}

with open(os.path.join(in_dir, "zh_nlpcc.tsv"), 'r', encoding='utf-8-sig') as r:
    raw_rows = list(csv.DictReader(r, delimiter='\t'))
exclude_labels = {'NONE', ''}
for row in filter(lambda row: row['STANCE'] not in exclude_labels, raw_rows):
    if row['STANCE'] not in label_map:
        pdb.set_trace()
    target = target_map[row['TARGET']]
    samples_by_target[target].append({
        "Context": row['TEXT'],
        "Target": target,
        "StanceType": "bi",
        "Stance": label_map[row['STANCE']],
        'Lang': "zh"
    })

random.seed(0)
train_rows = []
test_rows = []
for target in targets:
    target_samples = samples_by_target[target]
    random.shuffle(target_samples)
    splindex = int(.8 * len(target_samples))
    train_rows.extend(target_samples[:splindex])
    test_rows.extend(target_samples[splindex:])

def write_corpus(path, rows):
    with open(os.path.join(out_dir, path), 'w') as w:
        writer = csv.DictWriter(w,
                                fieldnames=['Context', 'Target', 'StanceType', 'Stance', 'Lang'],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
write_corpus("zh_nlpcc_train.csv", train_rows)
write_corpus("zh_nlpcc_test.csv", test_rows)