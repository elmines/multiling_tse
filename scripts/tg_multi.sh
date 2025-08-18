#!/bin/bash
ALL=${ALL:-0}
TRAIN_FT=${TRAIN_FT:-$ALL}

SEEDS=${@:- 0 112 343}

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-TGMulti}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME
LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"

if [ ! -e $LOGS_ROOT ]
then
    mkdir -p $LOGS_ROOT
fi

if [ $TRAIN_FT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse.train_ft \
            --seed $seed \
            --corpus_type li \
            -i data/li_tse/raw_train_all_onecol.csv \
            -o $LOGS_ROOT/ft_seed${seed}.model
    done
else
    echo "Skipping FastText training"
fi