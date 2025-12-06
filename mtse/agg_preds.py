import argparse
import pandas as pd
import pathlib

from .constants import TARGET_DELIMITER, C_SAMPLE, C_MAPPED_TARGET, C_UNTRANSLATED_TARGET, C_GENERATED_TARGET, C_LANG

# These are only for aggregated CSVs--shouldn't be used otherwise
C_ALL_UNTRANSLATED_TARGETS = "AllUntranslatedTargets"
C_ALL_GENERATED_TARGETS = "AllGeneratedTargets"

def make_string(df, col):
    df[col] = df[col].apply(str)

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=pathlib.Path, required=True)
    parser.add_argument('-o', type=pathlib.Path, required=True)

    parser.add_argument("--gen", type=pathlib.Path, required=False)
    parser.add_argument('--pred', type=pathlib.Path, required=False)
    parser.add_argument('--stance', type=pathlib.Path, required=False)
    args = parser.parse_args(raw_args)

    data_df = pd.read_csv(args.i, index_col=None)
    data_df.index.name = C_SAMPLE
    joined = data_df

    if args.gen:
        gen_df = pd.read_csv(args.gen, usecols=[C_SAMPLE, C_UNTRANSLATED_TARGET, C_GENERATED_TARGET], index_col=None)
        make_string(gen_df, C_UNTRANSLATED_TARGET)
        make_string(gen_df, C_GENERATED_TARGET)

        group_obj = gen_df.groupby(C_SAMPLE)

        grouped_gentarg = group_obj[C_GENERATED_TARGET].apply(TARGET_DELIMITER.join)
        grouped_gentarg.name = C_ALL_GENERATED_TARGETS

        grouped_utarg = group_obj[C_UNTRANSLATED_TARGET].apply(TARGET_DELIMITER.join)
        grouped_utarg.name = C_ALL_UNTRANSLATED_TARGETS

        joined = joined.join(grouped_utarg, how='inner').join(grouped_gentarg, how='inner')
    if args.pred:
        pred_df = pd.read_csv(args.pred, index_col=C_SAMPLE, usecols=[C_SAMPLE, C_UNTRANSLATED_TARGET, C_GENERATED_TARGET, C_MAPPED_TARGET])
        make_string(pred_df, C_UNTRANSLATED_TARGET)
        make_string(pred_df, C_GENERATED_TARGET)
        joined = joined.join(pred_df, how='inner')
    if args.stance:
        stance_df = pd.read_csv(args.stance, index_col=C_SAMPLE, usecols=[C_SAMPLE, "StancePred"])
        joined = joined.join(stance_df, how='inner')
    assert len(joined) == len(data_df)
    joined.to_csv(args.o, lineterminator='\n')


if __name__ == "__main__":
    main()