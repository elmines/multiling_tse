#!/bin/bash

DATA_DIR=$(dirname $0)/../data/kptimes

mkdir -p $DATA_DIR/trans_part
for lang in ca es et fr it zh
do
        python -m mtse.translate kptimes --lang $lang -i $DATA_DIR/en_part/${lang}_dev.jsonl -o $DATA_DIR/trans_part/${lang}_dev.jsonl
        python -m mtse.translate kptimes --lang $lang -i $DATA_DIR/en_part/${lang}_train.jsonl -o $DATA_DIR/trans_part/${lang}_train.jsonl
done