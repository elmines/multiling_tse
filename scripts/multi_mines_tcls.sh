#!/bin/bash
ALL=${ALL:-0}

FIT=${FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

SEEDS=${@:- 0 112 343}

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiMinesTCls}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

SCRUB_TARGETS=${SCRUB_TARGETS:-0}

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function v_train { echo MinesTClsOneShot_seed${1}; }


if [ $FIT -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/mines_tc_oneshot.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version $(v_train $seed) \
            --seed_everything $seed \
            $EXTRA_ARGS
    done
else
    echo "Skipping fitting"
fi

if [ $TARGET_TEST -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    for seed in $SEEDS
    do
        python -m mtse test \
            -c $LOGS_ROOT/$(v_train $seed)/config.yaml \
            --data configs/data/li_tc_test.yaml \
            --trainer.logger.version $(v_train $seed)_target_test \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes 19 \
            --ckpt_path $LOGS_ROOT/$(v_train $seed)/checkpoints/*ckpt \
            $EXTRA_ARGS
    done
else
    echo "Skipping target testing"
fi

if [ $STANCE_TEST -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    for seed in $SEEDS
    do
        # We override the existing callback because we're not testing TSE this time
        python -m mtse test \
            -c $LOGS_ROOT/$(v_train $seed)/config.yaml \
            --data configs/data/li_stance_test.yaml \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version $(v_train $seed)_stance_test \
            --ckpt_path $LOGS_ROOT/$(v_train $seed)/checkpoints/*ckpt \
            $EXTRA_ARGS
    done
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    # Scrub targets here
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_train $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --trainer.logger.version $(v_train)_tse_test \
            $EXTRA_ARGS
    done
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    # Scrub targets here
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_train $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --trainer.logger.version $(v_train)_tse_test_gt \
            --model.use_target_gt true \
            $EXTRA_ARGS
    done
else
    echo "Skipping gt tse testing"
fi