#!usr/bin/env python3
import csv
import random
import sys
import os
import re

def write_corpus(out_path, normalized_rows):
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w,
                                fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"],
                                lineterminator='\n'
                                )
        writer.writeheader()
        writer.writerows(normalized_rows)

def stance_filter(rows):
    against_samples = []
    favor_samples = []
    for row in rows:
        stance = row['Stance']
        if stance == 0:
            against_samples.append(row)
        else:
            assert stance == 1
            favor_samples.append(row)
    return against_samples, favor_samples

def split_rows_simple(row_set, K):
    fold_size = len(row_set) // K
    train_folds = []
    val_folds = []
    test_folds = []
    part_inds = []
    last_ind = 0
    for _ in range(K - 1):
        next_ind = last_ind + fold_size
        part_inds.append((last_ind, next_ind))
        last_ind = next_ind
    part_inds.append((last_ind, len(row_set)))
    # FIXME: Find a way to choose the val size more dynamically?
    val_size = len(row_set) // 10
    assert 0 < val_size < fold_size
    for (test_start, test_end) in part_inds:
        test_folds.append(row_set[test_start:test_end])
        nontest = row_set[:test_start] + row_set[test_end:]
        random.shuffle(nontest)
        val_folds.append(row_set[:val_size])
        train_folds.append(row_set[val_size:])
    return train_folds, val_folds, test_folds

def append_folds(root_train, root_val, root_test, sub_train, sub_val, sub_test):
    pairs = [
        (root_train, sub_train),
        (root_val, sub_val),
        (root_test, sub_test)
    ]
    for (root_folds, sub_folds) in pairs:
        for root_fold, class_fold in zip(root_folds, sub_folds):
            root_fold.extend(class_fold)

def split_rows_stance(row_set, K):
    against_samples, favor_samples = stance_filter(row_set)
    train_folds = [ [] for _ in range(K) ]
    val_folds = [ [] for _ in range(K) ]
    test_folds = [ [] for _ in range(K) ]
    for class_data in [against_samples, favor_samples]:
        class_train_folds, class_val_folds, class_test_folds = split_rows_simple(class_data, K)
        append_folds(train_folds, val_folds, test_folds, class_train_folds, class_val_folds, class_test_folds)
    return train_folds, val_folds, test_folds

def write_corpora(fold_dirs, train_folds, val_folds, test_folds, path_template):
    for fold_dir, train_fold, val_fold, test_fold in zip(fold_dirs, train_folds, val_folds, test_folds):
        for part, fold in [("train", train_fold), ("val", val_fold), ("test", test_fold)]:
            write_corpus(os.path.join(fold_dir, path_template.format(part=part)), fold)

def part_cic(in_dir, fold_dirs):
    K = len(fold_dirs)
    stance_map = {'AGAINST': 0, 'FAVOR': 1}
    URI_REGEX = re.compile(r'https://t.co/[a-zA-Z0-9]*')
    ca_rows = []
    es_rows = []
    for (lang, row_set) in [('ca', ca_rows), ('es', es_rows)]:
        for part in ['train', 'val', 'test']:
            in_path = os.path.join(in_dir, f"{lang}_catalonia_{part}.csv")
            with open(in_path, 'r') as r:
                raw_rows = list(csv.DictReader(r, delimiter='\t'))
            row_set.extend({
                "Context": URI_REGEX.sub("", row['TWEET']),
                "Target": "Catalonian Independence",
                "StanceType": "bi",
                "Stance": stance_map[row['LABEL']],
                "Lang": lang
            } for row in raw_rows if row['LABEL'] != 'NONE')
        random.shuffle(row_set)
        train_folds, val_folds, test_folds = split_rows_stance(row_set, K)
        write_corpora(fold_dirs, train_folds, val_folds, test_folds, lang + "_catalonia_{part}.csv")

def part_nlpcc(in_dir, fold_dirs):
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
        target = target_map[row['TARGET']]
        samples_by_target[target].append({
            "Context": row['TEXT'],
            "Target": target,
            "StanceType": "bi",
            "Stance": label_map[row['STANCE']],
            'Lang': "zh"
        })

    K = len(fold_dirs)
    root_train_folds = [[] for _ in range(K)]
    root_val_folds = [[] for _ in range(K)]
    root_test_folds = [[] for _ in range(K)]
    for target in targets:
        target_train_folds, target_val_folds, target_test_folds = split_rows_stance(samples_by_target[target], K)
        append_folds(root_train_folds, root_val_folds, root_test_folds,
                     target_train_folds, target_val_folds, target_test_folds)
    write_corpora(fold_dirs, root_train_folds, root_val_folds, root_test_folds, "zh_nlpcc_{part}.csv")



if __name__ == "__main__":
    random.seed(0)
    in_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "data", "multiling", "raw")
    out_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "data", "multiling")
    K = 5
    fold_dirs = []
    for i in range(K):
        fdir = os.path.join(out_dir, f"fold{i}")
        os.makedirs(fdir, exist_ok=True)
        fold_dirs.append(fdir)
    part_cic(in_dir, fold_dirs)
    part_nlpcc(in_dir, fold_dirs)

    