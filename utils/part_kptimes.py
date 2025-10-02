#!/usr/bin/env python3
import sys
import os
import random
import itertools

kp_root = os.path.join(os.path.dirname(sys.argv[0]), "..", "..", "data", "kptimes")

out_dir = os.path.join(kp_root, "en_part")
os.makedirs(out_dir, exist_ok=True)

langs = [
    "ca",
    "es",
    "et",
    "fr",
    "it",
    "zh"
]
ratios = [0.16, 0.16, 0.16, 0.16, 0.16, 0.17]


random.seed(0)
for source_path in ["dev.jsonl", "train.jsonl"]:
    with open(os.path.join(kp_root, source_path), 'r') as r:
        lines = list(itertools.islice([l for l in map(lambda line: line.strip(), r.readlines()) if l], 1000000))
    lang_assignments = random.choices(langs, weights=ratios, k=len(lines))
    lang_partitions = {l:[] for l in langs}
    for (line, lang) in zip(lines, lang_assignments):
        lang_partitions[lang].append(line)
    for lang, lang_lines in lang_partitions.items():
        with open(os.path.join(out_dir, f"{lang}_{source_path}"), 'w') as w:
            w.write("\n".join(lang_lines))
    