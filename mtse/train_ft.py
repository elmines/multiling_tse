"""
Train FastText embeddings
"""
# STL
import sys
import argparse
import pathlib
# 3rd Party
from gensim.models import FastText
from tokenizers.pre_tokenizers import BertPreTokenizer
# Local
from .data.parse import *

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_type', type=str, choices=list(CORPUS_PARSERS.keys()), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-i', type=pathlib.Path, required=True)
    parser.add_argument('-o', type=pathlib.Path, required=True)
    args = parser.parse_args(raw_args)

    parse_func = CORPUS_PARSERS[args.corpus_type]
    pretok = BertPreTokenizer()
    contexts = [ list(map(lambda x: x[0].lower(), pretok.pre_tokenize_str(s.context))) for s in parse_func(args.i)]
    model = FastText(vector_size=args.embed, seed=args.seed)
    model.build_vocab(corpus_iterable=contexts)
    model.train(corpus_iterable=contexts, total_examples=len(contexts), epochs=args.epochs)
    out_path = str(args.o)
    model.save(out_path)
    print(f"Saved FastText embeddings to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1:])