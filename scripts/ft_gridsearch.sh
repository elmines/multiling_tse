#!/bin/bash
EPOCHS=${EPOCHS:-10 100 500}
EMBED=${EMBED:-128 256 512}
SEEDS=${SEEDS:-0 112 343}

for epochs in $EPOCHS
do
    for embed in $EMBED
    do
        OUT_DIR=lightning_logs/FT_${epochs}epochs_${embed}embed
        mkdir -p $OUT_DIR
        for seed in $SEEDS
        do
            python -m mtse.train_ft \
                --corpus_type li \
                -i data/li_tse/raw_train_all_onecol.csv \
                --seed $seed \
                --embed $embed \
                --epochs $epochs \
                -o $OUT_DIR/ft_seed${seed}.model
        done
    done
done