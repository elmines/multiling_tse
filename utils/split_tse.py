#!/usr/bin/env python3
"""
Splits the MSE test corpus into files
representing individual corpora (SemEval, P-Stance, etc.)

Usage: python split_tse_test.py raw_test_all_onecol.csv out_dir/
"""
import os
import csv
import sys

ENCODING = 'ISO-8859-1'

def part_corpus(source_path, corpora_inds):
    with open(source_path, 'r', encoding=ENCODING) as r:
        reader = csv.DictReader(r)
        all_rows = list(reader)
    
    str2strance = {
        "AGAINST": 0,
        "FAVOR": 1,
        "NONE": 2,
        "Dummy Stance": 2
    }
    def convert_row(row):
        return {
            "Context": row["Tweet"],
            "Target": row["Target"],
            "StanceType": "tri",
            "Stance": str2strance[row["Stance"]],
            "Lang": "en"
        }
    all_rows = list(map(convert_row, all_rows))

    for filename, corpus_inds in corpora_inds.items():
        rowset = [all_rows[i] for i in corpus_inds]
        with open(filename, 'w', encoding=ENCODING) as w:
            writer = csv.DictWriter(w,
                                    fieldnames=["Context", "Target", "StanceType", "Stance", "Lang"],
                                    lineterminator='\n'
                                    )
            writer.writeheader()
            writer.writerows(rowset)

def extract_indices(corpus_path):
    semeval = []
    am = []
    covid = []
    ps = []
    unrel = []

    target_map = [
       ({"Joe Biden", "Bernie Sanders", "Donald Trump"}, ps) ,
       ({"Unrelated"}, unrel),
       ({'abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms'}, am),
       ({'face masks', 'fauci', 'stay at home orders', 'school closures'}, covid),
       ({"Hillary Clinton"}, semeval) # There's a SemEval sample without the hashtag but still has the Hillary Clinton target to identify it
    ]

    with open(corpus_path, 'r', encoding=ENCODING) as r:
        reader = csv.DictReader(r)
        for i, row in enumerate(reader):
            target = row['Target']
            if '#SemST' in row['Tweet']:
                semeval.append(i)
            else:
                _, index_arr = next(filter(lambda pair: target in pair[0], target_map))
                index_arr.append(i)
    return semeval, am, covid, ps, unrel


if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "data", "classic", "raw")
    out_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "data", "classic")

    data_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "data", "classic", "raw")
    add_dir_prefix = lambda slices: {os.path.join(out_dir, k):v for k,v in slices.items()}

    for part in ['val', 'train']:
        source_path = os.path.join(data_dir, f'raw_{part}_all_onecol.csv')
        corpora_inds = dict(zip(
            [f'semeval.{part}.csv', f'am.{part}.csv', f'covid.{part}.csv', f'pstance.{part}.csv', f'unrelated.{part}.csv'],
            extract_indices(source_path)
        ))
        part_corpus(source_path, add_dir_prefix(corpora_inds))

    # Use the exact indices that Li et al. (2023) used here, since we know them explicitly
    test_inds = {
       "semeval.test.csv": list(range(1080)),
       "am.test.csv": list(range(1880, 6989)),
       "covid.test.csv": list(range(1080, 1880)),
       "pstance.test.csv": list(range(6989, 9146)),
       "unrelated.test.csv": list(range(9146, 11026)) # This last one actually had to come up with manually
    }
    part_corpus(os.path.join(data_dir, f'raw_test_all_onecol.csv'), add_dir_prefix(test_inds))