# STL
import argparse
import pathlib
from typing import List
import itertools
import random
# 3rd Party
import torch
from transformers import pipeline, TranslationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.pipelines.base import pad_collate_fn
from tqdm import tqdm
# Local
from .utils import time_block
from .data import MapDataset
from .data.parse import *
from .data.target_pred import parse_target_preds, write_target_preds

def translate(model: M2M100ForConditionalGeneration,
              tokenizer: M2M100Tokenizer,
              texts: List[str],
              src_lang: str,
              tgt_lang: str = 'en',
              max_length: int = 512,
              batch_size: int = 32):

    dataset = MapDataset([
        tokenizer._build_translation_inputs(t,
                                            src_lang,
                                            tgt_lang,
                                            truncation=True,
                                            max_length=max_length,
                                            return_tensors='pt')
        for t in tqdm(texts, desc="Tokenizing text")
    ])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate_fn(tokenizer, feature_extractor=None)
    )
    output_texts = []
    for batch in tqdm(dataloader, desc="Translating batches"):
        batch = {k:(v.to(device=model.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        output_ids = model.generate(**batch, max_length=max_length)
        output_batch = [tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in output_ids]
        output_texts.extend(output_batch)
    return output_texts

def main(raw_args=None):
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

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

    kp_subparser = subparsers.add_parser("kptimes")
    kp_subparser.add_argument('-i', type=pathlib.Path, required=True)
    kp_subparser.add_argument('-o', type=pathlib.Path, required=True)
    kp_subparser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args(raw_args)

    def get_model():
        model_name = "facebook/m2m100_418M"
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device="cuda")
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    # TODO: Better way to check the subcommand ?
    if hasattr(args, "corpus_type"):
        in_paths = args.i
        out_paths = args.o
        assert len(in_paths) == len(out_paths)
        corpus_types = args.corpus_type
        assert len(corpus_types) == len(in_paths)
        model, tokenizer = get_model()
        for corpus_type, in_path, out_path in zip(corpus_types, in_paths, out_paths):
            parse_func = CORPUS_PARSERS[corpus_type]
            samples = list(tqdm(parse_func(in_path), desc=f"Parsing {in_path}"))
            src_lang = samples[0].lang
            assert src_lang
            assert all(s.lang == src_lang for s in samples)
            texts = [s.context for s in samples]
            translations = translate(model, tokenizer, texts, src_lang)
            for (s, translation) in zip(samples, translations):
                s.lang = 'en'
                s.context = translation
            write_standard(out_path, tqdm(samples, desc=f"Writing to {out_path}"))
    elif args.subcommand == "kptimes":
        model,tokenizer = get_model()
        texts = []
        labels = []
        sample_iter = parse_kptimes(args.i)
        for sample in sample_iter:
            texts.append(sample.context)
            labels.append(sample.target_label)

        lang = args.lang
        converted = []
        label_translations = translate(model, tokenizer, labels, src_lang='en', tgt_lang=lang)
        label_translations = [[l for l in map(lambda x: x.strip(), label_translation.split(TARGET_DELIMITER)) if l] for label_translation in label_translations]
        text_translations = translate(model, tokenizer, texts, src_lang='en', tgt_lang=lang)
        converted.extend( {"abstract": text, "keyphrases": label_group} for (text, label_group) in zip(text_translations, label_translations) )

        with open(args.o, 'w') as w:
            for entry in converted:
                w.write(json.dumps(entry) + "\n")
    else:
        in_paths = args.i
        out_paths = args.o
        assert len(in_paths) == len(out_paths)
        langs = args.lang
        assert len(langs) == len(in_paths)
        model, tokenizer = get_model()
        for src_lang, in_path, out_path in zip(langs, in_paths, out_paths):
            target_preds = list(tqdm(parse_target_preds(in_path), desc=f"Parsing {in_path}"))
            texts = []
            sample_inds: List[int] = []
            for i, pred in enumerate(target_preds):
                texts.extend(pred.generated_targets)
                sample_inds.extend(i for _ in pred.generated_targets)
                pred.generated_targets.clear()
            translations = translate(model, tokenizer, texts, src_lang)
            for (sample_ind, translation) in zip(sample_inds, translations):
                target_preds[sample_ind].generated_targets.append(translation)
            write_target_preds(out_path, tqdm(target_preds, desc=f"Writing to {out_path}"))


if __name__ == "__main__":
    main()