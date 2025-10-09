#!/usr/bin/env python3
import json
import csv
import random
import sys
import os
import re
from itertools import islice

UNREL_PERCENTAGE = 0.172
UNREL_RATIO = UNREL_PERCENTAGE / (1 - UNREL_PERCENTAGE)

def seed_and_shuffle(samples):
    random.seed(0)
    random.shuffle(samples)

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
    neut_samples = []
    for row in rows:
        stance = row['Stance']
        if stance == 0:
            against_samples.append(row)
        elif stance == 1:
            favor_samples.append(row)
        else:
            assert stance == 2
            neut_samples.append(row)
    return against_samples, favor_samples, neut_samples

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
        seed_and_shuffle(nontest)
        val_folds.append(nontest[:val_size])
        train_folds.append(nontest[val_size:])
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
    against_samples, favor_samples, neut_samples = stance_filter(row_set)
    train_folds = [ [] for _ in range(K) ]
    val_folds = [ [] for _ in range(K) ]
    test_folds = [ [] for _ in range(K) ]
    for class_data in [against_samples, favor_samples, neut_samples]:
        class_train_folds, class_val_folds, class_test_folds = split_rows_simple(class_data, K)
        append_folds(train_folds, val_folds, test_folds, class_train_folds, class_val_folds, class_test_folds)
    return train_folds, val_folds, test_folds

def write_corpora(fold_dirs, train_folds, val_folds, test_folds, path_template):
    for fold_dir, train_fold, val_fold, test_fold in zip(fold_dirs, train_folds, val_folds, test_folds):
        for part, fold in [("train", train_fold), ("val", val_fold), ("test", test_fold)]:
            write_corpus(os.path.join(fold_dir, path_template.format(part=part)), fold)

def part_cic(in_dir, fold_dirs):
    K = len(fold_dirs)
    stance_map = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
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
                "StanceType": "tri",
                "Stance": stance_map[row['LABEL']],
                "Lang": lang
            } for row in raw_rows)
        seed_and_shuffle(row_set)
        train_folds, val_folds, test_folds = split_rows_stance(row_set, K)
        write_corpora(fold_dirs, train_folds, val_folds, test_folds, lang + "_catalonia_{part}.csv")
    return len(ca_rows), len(es_rows)

def part_nlpcc(in_dir, fold_dirs):
    target_map = {
        "IphoneSE": "IphoneSE",
        "春节放鞭炮": "Setting off firecrackers during the Spring Festival",
        "俄罗斯在叙利亚的反恐行动": "Russia's counter-terrorism operations in Syria",
        "开放二胎": "Allowing second births",
        "深圳禁摩限电": "Shenzhen bans motorcycles and imposes electricity restrictions",
    }

    file_labels = {
        "IphoneSE": "iphone",
        "Setting off firecrackers during the Spring Festival": "firecracker",
        "Russia's counter-terrorism operations in Syria": "russia",
        "Allowing second births": "twochild",
        "Shenzhen bans motorcycles and imposes electricity restrictions": "shenzhen",
    }


    label_map = {
        'FAVOR': 1,
        'AGAINST': 0,
        'NONE': 2
    }
    targets = sorted(target_map.values())
    samples_by_target = {t:[] for t in targets}
    with open(os.path.join(in_dir, "zh_nlpcc.tsv"), 'r', encoding='utf-8-sig') as r:
        raw_rows = list(csv.DictReader(r, delimiter='\t'))
    bad_count = 0
    for row in raw_rows:
        if row['STANCE'] == '':
            bad_count += 1
            continue
        target = target_map[row['TARGET']]
        samples_by_target[target].append({
            "Context": row['TEXT'],
            "Target": target,
            "StanceType": "tri",
            "Stance": label_map[row['STANCE']],
            'Lang': "zh"
        })

    K = len(fold_dirs)
    for target in targets:
        target_samples = samples_by_target[target]
        seed_and_shuffle(target_samples)
        target_train_folds, target_val_folds, target_test_folds = split_rows_stance(target_samples, K)
        file_label = file_labels[target]
        write_corpora(fold_dirs, target_train_folds, target_val_folds, target_test_folds, f"zh_{file_label}_" + "{part}.csv")

    return sum(len(v) for v in samples_by_target.values())


def part_sardistance(in_dir, fold_dirs):
    label_map = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
    train_in_path       = os.path.join(in_dir, "it_sardinia_train.csv")
    test_in_path        = os.path.join(in_dir, "it_sardinia_test.csv")
    test_labels_in_path = os.path.join(in_dir, "it_sardinia_test_labels.csv")

    TARGET = "Sardinian Independence"

    with open(test_in_path, 'r') as r:
        raw_test_rows = list(csv.DictReader(r))
    with open(test_labels_in_path, 'r') as r:
        raw_test_label_rows = list(csv.DictReader(r))
    assert len(raw_test_rows) == len(raw_test_label_rows)
    normed_rows = []
    for (data, label) in zip(raw_test_rows, raw_test_label_rows):
        assert data['tweet_id'] == label['tweet_id']
        raw_label = label['label']    
        normed_rows.append({
            "Context": data['text'],
            "Target": TARGET,
            "Stance": label_map[raw_label],
            "StanceType": "tri",
            "Lang": "it"
        })
    with open(train_in_path, 'r') as r:
        raw_rows = list(csv.DictReader(r))
        for row in raw_rows:
            label = label_map[row['label']]
            normed_rows.append({
                "Context": row['text'],
                'Target': TARGET,
                'Stance': label,
                'StanceType': 'tri',
                'Lang': 'it'
            })
    seed_and_shuffle(normed_rows)
    train_folds, val_folds, test_folds = split_rows_stance(normed_rows, len(fold_dirs))
    write_corpora(fold_dirs, train_folds, val_folds, test_folds, "it_sardinia_{part}.csv")
    return len(normed_rows)

def part_et_data(in_dir, fold_dirs):
    in_path = os.path.join(in_dir, 'et_immigration.csv')
    label_map = {
        # Against
        '1': 0,
        '2': 0,
        # Neutral
        '3': 2,
        # Favor
        '4': 1,
        '5': 1,
        # Not related
        # 'MH': 2
    }
    with open(in_path, 'r') as r:
        raw_rows = list(csv.DictReader(r))
    rows = []
    unrel_rows = []
    for row in raw_rows:
        stance = row['stanceConsolidated']
        if stance in label_map:
            rows.append({
                "Context": row['sentence'],
                "Target": "Immigration",
                "Stance": label_map[stance],
                "StanceType": "tri",
                "Lang": "et"
            })
        else:
            assert stance == 'MH', stance
            unrel_rows.append({
                "Context": row['sentence'],
                "Target": "Unrelated",
                "Stance": 2,
                "StanceType": "tri",
                "Lang": "et"
            })
    n_unrel = int(len(rows) * UNREL_RATIO)
    assert len(unrel_rows) >= n_unrel
    seed_and_shuffle(unrel_rows)
    unrel_rows = unrel_rows[:n_unrel]
    unrel_train, unrel_val, unrel_test = split_rows_simple(unrel_rows, len(fold_dirs))
    write_corpora(fold_dirs, unrel_train, unrel_val, unrel_test, "et_unrelated_{part}.csv")
    

    seed_and_shuffle(rows)
    train_folds, val_folds, test_folds = split_rows_stance(rows, len(fold_dirs))
    write_corpora(fold_dirs, train_folds, val_folds, test_folds, "et_immigration_{part}.csv")


def part_fr_election_data(in_dir, fold_dirs):
    URI_REGEX = re.compile(r'https://t.co/[a-zA-Z0-9]*')

    entries = [
        ("fr_lepen.csv", "fr_lepen_{part}.csv", "Marine LePen"),
        ("fr_macron.csv", "fr_macron_{part}.csv", "Emmanuel Macron"),
    ]
    label_map = {
        'favor': 1,
        'FAVOUR': 1,
        'agains': 0,
        'AGAINST': 0,
        'none': 2,
        'NONE': 2
    }
    row_count = 0
    for (in_name, path_template, target) in entries:
        with open(os.path.join(in_dir, in_name), 'r') as r:
            raw_rows = list(csv.DictReader(r))
        normed_rows = [{
            "Context": URI_REGEX.sub("", row['Tweet']),
            "Target": target,
            "Stance": label_map[row['Stance']],
            "StanceType": "tri",
            "Lang": "fr"
        } for row in raw_rows]
        row_count += len(normed_rows)
        seed_and_shuffle(normed_rows)
        train_folds, val_folds, test_folds = split_rows_stance(normed_rows, len(fold_dirs))
        write_corpora(fold_dirs, train_folds, val_folds, test_folds, path_template)
    return row_count

def part_cstance_data(in_dir, fold_dirs, n_samples):
    in_path = os.path.join(in_dir, 'zh_cstance.csv')
    with open(in_path, 'r', encoding='utf-8-sig') as r:
        raw_rows = list(csv.DictReader(r))
    assert len(raw_rows) >= n_samples
    seed_and_shuffle(raw_rows)
    raw_rows = raw_rows[:n_samples]
    rows = [{"Context": raw_row['Text'],
            "Target": "Unrelated",
            'StanceType': 'tri',
            "Stance": 2, # Neutral
            'Lang': 'zh'} for raw_row in raw_rows]
    train_folds, val_folds, test_folds = split_rows_simple(rows, len(fold_dirs))
    write_corpora(fold_dirs, train_folds, val_folds, test_folds, "zh_unrelated_{part}.csv")

def part_enc_data(in_dir, fold_dirs):
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
    seed_and_shuffle(samples)
    samples = samples[:n_samples]
    rows = [
        {"Context": t, "Target": "Unrelated", "StanceType": "tri", "Stance": 2, "Lang": 'et'}
        for t in samples
    ]
    train_folds, val_folds, test_folds = split_rows_simple(rows, len(fold_dirs))
    write_corpora(fold_dirs, train_folds, val_folds, test_folds, "et_unrelated_{part}.csv")

def texts_to_samples(lang, texts):
    return [
        {"Context": t, "Target": "Unrelated", "StanceType": "tri", "Stance": 2, "Lang": lang}
        for t in texts
    ]

MIN_GV_CHARS = 128
"""
Minimum chars for GlobalVoices
"""

def part_global_voices(in_dir, fold_dirs, lang, n_samples):
    in_path = os.path.join(in_dir, f"{lang}_globalvoices.txt")
    with open(in_path, 'r') as r:
        samples = [l.strip() for l in r.readlines()]
    samples = list(filter(lambda l: len(l) >= MIN_GV_CHARS, samples))
    assert len(samples) >= n_samples
    seed_and_shuffle(samples)
    samples = samples[:n_samples]
    samples = texts_to_samples(lang, samples)
    train_folds, val_folds, test_folds = split_rows_simple(samples, len(fold_dirs))
    write_corpora(fold_dirs, train_folds, val_folds, test_folds, lang + "_unrelated_{part}.csv")

def get_en_globalvoices(in_dir, out_dir, n_samples=64000):
    """
    The Li et al. in their TSE paper used ~51,000 samples to train their FT
    embeddings; we use 64000 just to be safe
    """
    # English is just for the embedding training
    in_path = os.path.join(in_dir, f"en_globalvoices.txt")
    with open(in_path, 'r') as r:
        samples = [l.strip() for l in r.readlines()]
    samples = list(filter(lambda l: len(l) >= MIN_GV_CHARS, samples))
    assert len(samples) >= n_samples
    seed_and_shuffle(samples)
    samples = samples[:n_samples]
    samples = texts_to_samples('en', samples)
    write_corpus(os.path.join(out_dir, "en_unrelated.csv"), samples)


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

    # et subroutine already takes care of Unrelated data
    part_et_data(in_dir, fold_dirs)
    ca_count, es_count = part_cic(in_dir, fold_dirs)
    fr_count = part_fr_election_data(in_dir, fold_dirs)
    zh_count = part_nlpcc(in_dir, fold_dirs)
    it_count = part_sardistance(in_dir, fold_dirs)

    part_global_voices(in_dir, fold_dirs, 'ca', int(ca_count * UNREL_RATIO))
    part_global_voices(in_dir, fold_dirs, 'es', int(es_count * UNREL_RATIO))
    part_global_voices(in_dir, fold_dirs, 'it', int(it_count * UNREL_RATIO))
    part_global_voices(in_dir, fold_dirs, 'fr', int(fr_count * UNREL_RATIO))
    part_cstance_data(in_dir, fold_dirs, int(zh_count * UNREL_RATIO))

    # Unrelated Data
    # part_enc_data(in_dir, fold_dirs)
    # part_cstance_data(in_dir, fold_dirs)
    # part_globalvoices_data(in_dir, fold_dirs, out_dir)
    # Core Data

    