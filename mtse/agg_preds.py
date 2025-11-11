import argparse
import pandas as pd
import pathlib

from .constants import TARGET_DELIMITER

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=pathlib.Path, required=True)
    parser.add_argument('-o', type=pathlib.Path, required=True)

    parser.add_argument("--gen", type=pathlib.Path, required=False)
    parser.add_argument('--pred', type=pathlib.Path, required=False)
    parser.add_argument('--stance', type=pathlib.Path, required=False)
    args = parser.parse_args(raw_args)

    data_df = pd.read_csv(args.i, index_col=None)
    data_df.index.name = 'Sample'
    joined = data_df
    if args.gen:
        gen_df = pd.read_csv(args.gen, usecols=["Sample", "Generated Target"], index_col=None)
        gen_df = gen_df.groupby("Sample")['Generated Target'].apply(TARGET_DELIMITER.join)
        gen_df.name = "TargetCandidates"
        joined = joined.join(gen_df, how='inner')
    if args.pred:
        pred_df = pd.read_csv(args.pred, index_col='Sample', usecols=["Sample", "Generated Target", "Mapped Target"])
        pred_df.rename(columns={"Generated Target": "GeneratedTarget", "Mapped Target": "MappedTarget"}, inplace=True)
        joined = joined.join(pred_df, how='inner')
    if args.stance:
        stance_df = pd.read_csv(args.stance, index_col='Sample', usecols=["Sample", "StancePred"])
        joined = joined.join(stance_df, how='inner')
    assert len(joined) == len(data_df)
    joined.to_csv(args.o, lineterminator='\n')


if __name__ == "__main__":
    main()