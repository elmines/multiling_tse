"""
Splits the MSE test corpus into files
representing individual corpora (SemEval, P-Stance, etc.)

Usage: python split_tse_test.py raw_test_all_onecol.csv out_dir/
"""
import os
import csv

ENCODING = 'ISO-8859-1'

def part_corpus(source_path, corpus_slices):
    with open(source_path, 'r', encoding=ENCODING) as r:
        reader = csv.DictReader(r)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    for filename, corpus_slice in corpus_slices.items():
        rowset = all_rows[corpus_slice]
        with open(filename, 'w', encoding=ENCODING) as w:
            writer = csv.DictWriter(w, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rowset)


if __name__ == "__main__":

    data_dir = os.path.join("data", "li_tse")
    add_dir_prefix = lambda slices: {os.path.join(data_dir, k):v for k,v in slices.items()}

    test_slices = {
        "test_semeval.csv": slice(None, 1080),
        "test_covid.csv": slice(1080,1880),
        "test_am.csv": slice(1880,6989),
        "test_pstance.csv": slice(6989,9146),
        "test_unrelated.csv": slice(9146, None)
    }
    part_corpus(os.path.join(data_dir, 'raw_test_all_onecol.csv'), add_dir_prefix(test_slices))