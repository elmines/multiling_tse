#!/bin/bash

DATA_DIR=$(dirname $0)/../data/kptimes

mkdir -p $DATA_DIR/trans_part
for lang in ca es et fr it zh
do
        python -m mtse.translate kptimes --lang $lang -i data/kptimes/en_part/${lang}_train.jsonl -o data/kptimes/trans_part/${lang}_train.jsonl
        python -m mtse.translate kptimes --lang $lang -i data/kptimes/en_part/${lang}_dev.jsonl -o data/kptimes/trans_part/${lang}_dev.jsonl
done