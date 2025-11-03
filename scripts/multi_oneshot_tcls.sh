#!/bin/bash
ALL=${ALL:-0}

FIT=${FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

seed=${1:-0}
SCRUB_TARGETS=${SCRUB_TARGETS:-0}

if [ $SCRUB_TARGETS -eq 1 ]
then
    DEFAULT_EXP_NAME=MultiOneshotTClsWithScrub
else
    DEFAULT_EXP_NAME=MultiOneshotTCls
fi

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-$DEFAULT_EXP_NAME}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


if [ $FIT -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

        python -m mtse fit \
            -c configs/base/oneshot_tcls.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed} \
            --seed_everything $seed \
            $EXTRA_ARGS
else
    echo "Skipping fitting"
fi

if [ $TARGET_TEST -eq 1 ]
then
        python -m mtse test \
            -c $LOGS_ROOT/seed${seed}/config.yaml \
            --trainer.logger.version seed${seed}_target_test \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes 19 \
            --ckpt_path $LOGS_ROOT/seed${seed}/checkpoints/*ckpt
else
    echo "Skipping target testing"
fi

if [ $STANCE_TEST -eq 1 ]
then
        # We override the existing callback because we're not testing TSE this time
        python -m mtse test \
            -c $LOGS_ROOT/seed${seed}/config.yaml \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version seed${seed}_stance_test \
            --ckpt_path $LOGS_ROOT/seed${seed}/checkpoints/*ckpt \
            $EXTRA_ARGS
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
        train_dir=$LOGS_ROOT/seed${seed}
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --trainer.logger.version seed${seed}_tse_test
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
        train_dir=$LOGS_ROOT/seed${seed}
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --trainer.logger.version seed${seed}_tse_test_gt \
            --model.use_target_gt true
else
    echo "Skipping gt tse testing"
fi