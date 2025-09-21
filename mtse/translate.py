# STL
import pdb
import argparse
import pathlib
from typing import List, Optional
# 3rd Party
from transformers import pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
# Local
from .data.parse import *
from .data.target_pred import parse_target_preds, write_target_preds

def translate(translator: pipeline, texts: List[str], src_lang: str, pbar: Optional[tqdm] = None):
    if pbar is None:
        with tqdm(total=len(texts), desc=f"Translating") as pbar:
            translate(translator, texts, src_lang, pbar)

    # TODO: Less crude way of making single column dataset?
    dataset = KeyDataset(
        Dataset.from_dict({"text": texts}),
        "text"
    )
    translations = []
    for res in translator(dataset, src_lang=src_lang, tgt_lang='en'):
        translations.extend(res)
        pbar.update()
    return [t['translation_text'] for t in translations]

def main(raw_args=None):
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="Subcommands")

    # TODO: Add these to the root parser, if I can do it correctly...
    def add_common_args(subparser):
        subparser.add_argument('-i', nargs="+", type=pathlib.Path, required=True)
        subparser.add_argument('-o', nargs="+", type=pathlib.Path, required=True)

    stance_subparser = subparsers.add_parser("stance")
    add_common_args(stance_subparser)
    stance_subparser.add_argument('--corpus_type', nargs="+", type=str, choices=list(CORPUS_PARSERS.keys()), required=True)

    pred_subparser = subparsers.add_parser("pred")
    add_common_args(pred_subparser)
    pred_subparser.add_argument("--lang", nargs="+", type=str, required=True)

    args = parser.parse_args(raw_args)

    batch_size = 32
    in_paths = args.i
    out_paths = args.o
    assert len(in_paths) == len(out_paths)

    def get_translator():
        return pipeline("translation",
                              model="facebook/m2m100_418M",
                              batch_size=batch_size,
                              truncation=True,
                              max_length=500)
    # TODO: Better way to check the subcommand ?
    if hasattr(args, "corpus_type"):
        corpus_types = args.corpus_type
        assert len(corpus_types) == len(in_paths)
        translator = get_translator()
        for corpus_type, in_path, out_path in zip(corpus_types, in_paths, out_paths):
            parse_func = CORPUS_PARSERS[corpus_type]
            samples = list(tqdm(parse_func(in_path), desc=f"Parsing {in_path}"))
            src_lang = samples[0].lang
            assert src_lang
            assert all(s.lang == src_lang for s in samples)
            texts = [s.context for s in samples]
            translations = translate(translator, texts, src_lang)
            for (s, translation) in zip(samples, translations):
                s.lang = 'en'
                s.context = translation
            write_standard(out_path, tqdm(samples, desc=f"Writing to {out_path}"))
    else:
        langs = args.lang
        assert len(langs) == len(in_paths)
        translator = get_translator()
        for src_lang, in_path, out_path in zip(langs, in_paths, out_paths):
            target_preds = list(tqdm(parse_target_preds(in_path), desc=f"Parsing {in_path}"))
            texts = []
            sample_inds: List[int] = []
            for i, pred in enumerate(target_preds):
                texts.extend(pred.generated_targets)
                sample_inds.extend(i for _ in pred.generated_targets)
                pred.generated_targets.clear()
            translations = translate(translator, texts, src_lang)
            for (sample_ind, translation) in zip(sample_inds, translations):
                target_preds[sample_ind].generated_targets.append(translation)
            write_target_preds(out_path, tqdm(target_preds, desc=f"Writing to {out_path}"))


if __name__ == "__main__":
    main()