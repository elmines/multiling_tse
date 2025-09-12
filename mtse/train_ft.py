"""
Train FastText embeddings
"""
# STL
import sys
import argparse
import pathlib
import logging
# 3rd Party
from gensim.models import FastText
from tokenizers.pre_tokenizers import BertPreTokenizer
from tqdm import tqdm
# Local
from .data.parse import *

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_type', nargs="+", type=str, choices=list(CORPUS_PARSERS.keys()), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-i', nargs="+", type=pathlib.Path, required=True)
    parser.add_argument('-o', type=pathlib.Path, required=True)
    args = parser.parse_args(raw_args)

    logging.basicConfig(level=logging.INFO)

    assert len(args.corpus_type) == len(args.i)
    pretok = BertPreTokenizer()

    contexts = []
    for corpus_type, in_path in zip(args.corpus_type, args.i):
        parse_func = CORPUS_PARSERS[corpus_type]
        sample_iter = tqdm(parse_func(in_path), desc=f"Parsing {in_path}")
        contexts.extend(list(map(lambda x: x[0].lower(), pretok.pre_tokenize_str(s.context))) for s in sample_iter)
    model = FastText(vector_size=args.embed, seed=args.seed)
    model.build_vocab(corpus_iterable=contexts)
    model.train(
        corpus_iterable=contexts,
        total_examples=len(contexts),
        epochs=args.epochs,
    )
    out_path = str(args.o)
    model.save(out_path)
    print(f"Saved FastText embeddings to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1:])