#!/usr/bin/env

import pathlib
import argparse
from itertools import batched

from .data.target_pred import parse_target_preds, TargetPred

from lingua import Language, LanguageDetectorBuilder, IsoCode639_1

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=pathlib.Path)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args(raw_args)
    in_path = args.i
    batch_size = args.batch

    detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.CATALAN,
        Language.SPANISH,
        Language.ESTONIAN,
        Language.FRENCH,
        Language.ITALIAN,
        Language.CHINESE
    ).build()

    def map_target(pred: TargetPred):
        assert len(pred.untranslated_targets) == 1
        assert pred.lang is not None
        iso_code = IsoCode639_1.from_str(pred.lang.upper())
        lang = Language.from_iso_code_639_1(iso_code)
        return (pred.untranslated_targets[0], lang)

    target_iter = parse_target_preds(in_path)
    target_iter = map(map_target, target_iter)

    correct = 0
    total = 0
    for batch in batched(target_iter, batch_size):
        x_batch = [p[0] for p in batch]
        gt_batch = [p[1] for p in batch]
        pred_batch = detector.detect_languages_in_parallel_of(x_batch)
        correct += sum(gt == pred for gt,pred in zip(gt_batch, pred_batch))
        total += len(gt_batch)
    print(correct, total)


if __name__ == "__main__":
    main()
    pass