"""
Splits the MSE test corpus into files
representing individual corpora (SemEval, P-Stance, etc.)

Usage: python split_tse_test.py raw_test_all_onecol.csv out_dir/
"""
import sys
import os
import csv

def main(in_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(in_path, 'r') as r:
        reader = csv.DictReader(r)
        fieldnames = reader.fieldnames
        rows = list(reader)
    corpus_rows = {
        "test_semeval.csv": rows[:1080],
        "test_covid.csv": rows[1080:1880],
        "test_am.csv": rows[1880:6989],
        "test_pstance.csv": rows[6989:9146]
    }
    for (filename, rowset) in corpus_rows.items():
        with open(os.path.join(out_dir, filename), 'w') as w:
            writer = csv.DictWriter(w, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rowset)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])