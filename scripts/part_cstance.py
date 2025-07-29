import os
import random
import csv
import sys

# Right now these partitions are intended to be paired with
# part_nlpcc's partitions
# 600 samples for each of 5 targets
NLPCC_TARGET_SAMPLES = 600

def main(source_path, dest_dir, seed=0, k=5):
    assert NLPCC_TARGET_SAMPLES % k == 0
    base_test_size = int(1/k * NLPCC_TARGET_SAMPLES)
    os.makedirs(dest_dir, exist_ok=True)
    with open(source_path, 'r', encoding='utf-8-sig') as r:
        reader = csv.DictReader(r, delimiter=',')
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert len(rows) >= NLPCC_TARGET_SAMPLES
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:NLPCC_TARGET_SAMPLES]

    def write_corpus(path, samples):
        with open(path, 'w', encoding='utf-8-sig') as w:
            writer = csv.DictWriter(w,
                                    fieldnames=fieldnames,
                                    delimiter=',',
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(samples)

    train_folds = {}
    test_folds = {}
    for i in range(k):
        test_start = i * base_test_size
        test_end = (i + 1) * base_test_size
        train_folds[i] = rows[:test_start] + rows[test_end:]
        test_folds[i] = rows[test_start:test_end]
    for i in range(k):
        fold_dir = os.path.join(dest_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        write_corpus(os.path.join(fold_dir, 'train.tsv'), train_folds[i])
        write_corpus(os.path.join(fold_dir, 'test.tsv'), test_folds[i])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
    main(*sys.argv[1:])