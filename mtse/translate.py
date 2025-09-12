# STL
import argparse
import pathlib
# 3rd Party
from transformers import pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
# Local
from .data.parse import *

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_type', nargs="+", type=str, choices=list(CORPUS_PARSERS.keys()), required=True)
    parser.add_argument('-i', nargs="+", type=pathlib.Path, required=True)
    parser.add_argument('-o', nargs="+", type=pathlib.Path, required=True)
    args = parser.parse_args(raw_args)

    batch_size = 32
    corpus_types = args.corpus_type
    in_paths = args.i
    out_paths = args.o
    assert len(corpus_types) == len(in_paths)
    assert len(in_paths) == len(out_paths)

    translator = pipeline("translation",
                          model="facebook/m2m100_418M",
                          batch_size=batch_size,
                          truncation=True,
                          max_length=500
                          )

    for corpus_type, in_path, out_path in zip(corpus_types, in_paths, out_paths):
        parse_func = CORPUS_PARSERS[corpus_type]
        samples = list(tqdm(parse_func(in_path), desc=f"Parsing {in_path}"))
        src_lang = samples[0].lang
        assert src_lang
        assert all(s.lang == src_lang for s in samples)

        # TODO: Less crude way of making single column dataset?
        dataset = KeyDataset(
            Dataset.from_dict({"text": [s.context for s in samples]}),
            "text"
        )
        translations = []
        with tqdm(total=len(samples), desc=f"Translating {in_path}") as pbar:
            for res in translator(dataset, src_lang=src_lang, tgt_lang='en'):
                translations.extend(res)
                pbar.update()
        for (s, translation) in zip(samples, translations):
            s.lang = 'en'
            s.context = translation['translation_text']
        write_standard(out_path, tqdm(samples, desc=f"Writing to {out_path}"))

if __name__ == "__main__":
    main()