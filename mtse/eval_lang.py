#!/usr/bin/env

import sys
import csv
import os
import pathlib
import argparse
import glob
from itertools import batched

from .data.target_pred import parse_target_preds, TargetPred

from lingua import Language, LanguageDetectorBuilder

def lang_to_str(l: Language):
    return l.iso_code_639_1.name.lower()

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=pathlib.Path)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args(raw_args)
    in_dir = args.i
    batch_size = args.batch

    EXPECTED_LANGS = [
        Language.CATALAN,
        Language.SPANISH,
        Language.ESTONIAN,
        Language.FRENCH,
        Language.ITALIAN,
        Language.CHINESE
    ]
    detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, *EXPECTED_LANGS).build()
    EXPECTED_LANGS = [lang_to_str(l) for l in EXPECTED_LANGS]

    def target_iter(in_path):
        for pred in parse_target_preds(in_path):
            assert pred.lang is not None
            lang = pred.lang
            assert pred.untranslated_targets
            for targ in pred.untranslated_targets:
                yield targ, lang

    fold_dirs = glob.glob(os.path.join(in_dir, "fold*target_gen"))
    assert len(fold_dirs) == args.k

    fold_results = []
    for fold_dir in fold_dirs:
        in_paths = glob.glob(os.path.join(fold_dir, "*.target_gens.csv"))
        lang_correct = {l:0 for l in EXPECTED_LANGS}
        lang_total = {l:0 for l in EXPECTED_LANGS}

        for in_path in in_paths:
            for batch in batched(target_iter(in_path), batch_size):
                x_batch = [p[0] for p in batch]
                gt_batch = [p[1] for p in batch]
                pred_batch = detector.detect_languages_in_parallel_of(x_batch)
                pred_batch = [lang_to_str(l) if l is not None else None for l in pred_batch]
                for p, gt in zip(pred_batch, gt_batch):
                    lang_correct[gt] += (p == gt)
                    lang_total[gt] += 1
        result_set = {
            l : lang_correct[l] / lang_total[l] for l in EXPECTED_LANGS
        }
        result_set['all'] = sum(lang_correct.values()) / sum(lang_total.values())
        fold_results.append(result_set)

    mean_results = {}
    col_names = EXPECTED_LANGS + ["all"]
    for l in col_names:
        mean_results[l] = sum(res_set[l] for res_set in fold_results) / len(fold_results)

    writer = csv.DictWriter(sys.stdout, fieldnames=col_names, lineterminator='\n')
    writer.writeheader()
    writer.writerow(mean_results)
    


if __name__ == "__main__":
    main()