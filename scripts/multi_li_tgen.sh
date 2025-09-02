#!/bin/bash
ALL=${ALL:-0}
FT_EMBED=${FT_EMBED:-$ALL}
TARGET_FIT=${TARGET_FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
TARGET_PRED=${TARGET_PRED:-$ALL}
STANCE_FIT=${STANCE_FIT:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

SEEDS=${@:- 0 1 2}


SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiLiTGen}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function v_target { echo target_seed${1}; }

function v_target_predict { echo $($v_target $1)_predict; }

function v_stance { echo stance_seed${1}; }

function embed_path { echo $LOGS_ROOT/ft_seed${seed}.model; }

if [ $FT_EMBED -eq 1 ]
then
    mkdir -p $LOGS_ROOT
    for seed in $SEEDS
    do
        python -m mtse.train_ft \
            --corpus_type li \
            -i data/li_tse/raw_train_all_onecol.csv \
            --seed $seed \
            --embed 256 \
            -o $(embed_path $seed) \
            --epochs 500 
    done
else
    echo "Skipping FastText embedding"
fi

if [ $TARGET_FIT -eq 1 ]
then
    # FIXME: Use the proper embeddings path
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/li_target_generator.yaml \
            --model.embeddings_path /home/ethanlmines/blue_dir/lightning_logs/MultiMinesTGen/ft_seed0.model \
            $LOGGER_ARGS \
            --trainer.logger.version $(v_target $seed) \
            --seed_everything $seed \
            --trainer.max_time 00:00:01:00
    done
else
    echo "Skipping target fitting"
fi

if [ $TARGET_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse test \
            -c $LOGS_ROOT/$(v_target $seed)/config.yaml \
            --data configs/data/li_tc_test.yaml \
            --trainer.logger.version $(v_target $seed)_test \
            --ckpt_path $LOGS_ROOT/$(v_target $seed)/checkpoints/*ckpt \
            $EXTRA_ARGS
    done
else
    echo "Skipping target testing"
fi

if [ $TARGET_PRED -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse predict \
            -c $LOGS_ROOT/$(v_target $seed)/config.yaml \
            --data configs/data/li_tc_predict.yaml \
            --trainer.logger.version $(v_target_predict $seed) \
            --ckpt_path $LOGS_ROOT/$(v_target $seed)/checkpoints/*ckpt
    done
else
    echo "Skipping target prediction"
fi

if [ $STANCE_FIT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/li_stance_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version $(v_stance $seed) \
            --seed_everything $seed
    done
else
    echo "Skipping stance fitting"
fi

if [ $STANCE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        # We override the existing callback because we're not testing TSE this time
        python -m mtse test \
            -c $LOGS_ROOT/$(v_stance $seed)/config.yaml \
            --data configs/data/li_stance_test.yaml \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version $(v_stance $seed)_test \
            --ckpt_path $LOGS_ROOT/$(v_stance $seed)/checkpoints/*ckpt
    done
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_stance $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --data.corpora.target_preds_path $LOGS_ROOT/$(v_target_predict $seed)/target_preds.1.txt \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version seed${seed}_tse_test
    done
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_stance $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version seed${seed}_tse_test_gt \
            --data.corpora.target_input label \
            --model.use_target_gt true
    done
else
    echo "Skipping gt tse testing"
fi