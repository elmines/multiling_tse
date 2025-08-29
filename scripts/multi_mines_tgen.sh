#!/bin/bash
ALL=${ALL:-0}

FT_EMBED=${FT_EMBED:-$ALL}
FIT=${FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

SEEDS=${@:- 0 1 2}

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiMinesTGen}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function v_train { echo MinesTGen_seed${1}; }

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

if [ $FIT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/mines_tg_oneshot.yaml \
            --model.embeddings_path $(embed_path $seed) \
            $LOGGER_ARGS \
            --trainer.logger.version $(v_train $seed) \
            --seed_everything $seed \
            --trainer.max_epochs 1
    done
else
    echo "Skipping fitting"
fi

if [ $TARGET_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse test \
            -c $LOGS_ROOT/$(v_train $seed)/config.yaml \
            --data configs/data/li_tc_test.yaml \
            --trainer.logger.version $(v_train $seed)_target_test \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes $((1 + $(wc -l < static/li_merged_targets.txt) )) \
            --ckpt_path $LOGS_ROOT/$(v_train $seed)/checkpoints/*ckpt 
    done
else
    echo "Skipping target testing"
fi

if [ $STANCE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        # We override the existing callback because we're not testing TSE this time
        python -m mtse test \
            -c $LOGS_ROOT/$(v_train $seed)/config.yaml \
            --data configs/data/li_stance_test.yaml \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version $(v_train $seed)_stance_test \
            --ckpt_path $LOGS_ROOT/$(v_train $seed)/checkpoints/*ckpt 
    done
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_train $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --trainer.logger.version $(v_train)_tse_test 
    done
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_train $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --trainer.logger.version $(v_train)_tse_test_gt \
            --model.use_target_gt true 
    done
else
    echo "Skipping gt tse testing"
fi