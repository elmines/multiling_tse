"""
Usage: python part_nlpcc.py /path/to/evasampledata4-TaskAA.txt <output_dir>

Partitions the NLPCC training data
"""
import sys
import random
import os
import csv
from collections import defaultdict

def main(source_path, dest_dir, seed=0, k=5):
    os.makedirs(dest_dir, exist_ok=True)

    samples_by_target = defaultdict(list)
    with open(source_path, 'r') as r:
        reader = csv.DictReader(r, delimiter='\t')
        columns = reader.fieldnames
        for row in reader:
            samples_by_target[row['TARGET']].append(row)
    samples_by_target = dict(samples_by_target)

    rng = random.Random(seed)
    for samples in samples_by_target.values():
        rng.shuffle(samples)

    train_samples_by_fold = {i:[] for i in range(k)}
    test_samples_by_fold = {i:[] for i in range(k)}
    for samples in samples_by_target.values():
        assert len(samples) % k == 0
        chunk_size = len(samples) // k
        for i_fold in range(k):
            test_start = i_fold*chunk_size
            test_end = (i_fold + 1)*chunk_size
            train_samples_by_fold[i_fold].extend(samples[:test_start])
            test_samples_by_fold[i_fold].extend(samples[test_start:test_end])
            train_samples_by_fold[i_fold].extend(samples[test_end:])
    def write_corpus(path, samples):
        with open(path, 'w') as w:
            writer = csv.DictWriter(w, fieldnames=columns, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(samples)
    for i_fold in range(k):
        fold_dir = os.path.join(dest_dir, f"fold_{i_fold}")
        os.makedirs(fold_dir)
        write_corpus(os.path.join(fold_dir, 'train.tsv'), train_samples_by_fold[i_fold])
        write_corpus(os.path.join(fold_dir, 'test.tsv'), test_samples_by_fold[i_fold])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
    main(*sys.argv[1:])