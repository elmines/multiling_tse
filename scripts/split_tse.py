"""
Splits the MSE test corpus into files
representing individual corpora (SemEval, P-Stance, etc.)

Usage: python split_tse_test.py raw_test_all_onecol.csv out_dir/
"""
import os
import csv

ENCODING = 'ISO-8859-1'

def copy_se_rows(source_path, out_path):
    with open(source_path, 'r', encoding=ENCODING) as r:
        reader = csv.DictReader(r)
        fieldnames = reader.fieldnames
        train_rows = list(reader) 
    se_trainrows = [row for row in train_rows if '#SemST' in row['Tweet']]
    with open(out_path, 'w', encoding=ENCODING) as w:
        writer = csv.DictWriter(w, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(se_trainrows)

if __name__ == "__main__":

    data_dir = os.path.join("data", "li_tse")

    with open(os.path.join(data_dir, "raw_test_all_onecol.csv"), 'r', encoding=ENCODING) as r:
        reader = csv.DictReader(r)
        fieldnames = reader.fieldnames
        test_rows = list(reader)
    corpus_rows = {
        "test_semeval.csv": test_rows[:1080],
        "test_covid.csv": test_rows[1080:1880],
        "test_am.csv": test_rows[1880:6989],
        "test_pstance.csv": test_rows[6989:9146],
        "test_unrelated.csv": test_rows[9146:]
    }
    for (filename, rowset) in corpus_rows.items():
        with open(os.path.join(data_dir, filename), 'w', encoding=ENCODING) as w:
            writer = csv.DictWriter(w, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rowset)

    copy_se_rows(os.path.join(data_dir, 'raw_train_all_onecol.csv'), os.path.join(data_dir, 'train_semeval.csv'))
    copy_se_rows(os.path.join(data_dir, 'raw_val_all_onecol.csv'), os.path.join(data_dir, 'val_semeval.csv'))
