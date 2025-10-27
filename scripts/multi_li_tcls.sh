#!/bin/bash
ALL=${ALL:-0}
TARGET_FIT=${TARGET_FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
TARGET_PRED=${TARGET_PRED:-$ALL}
STANCE_FIT=${STANCE_FIT:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

seed=${1:- 0}

WITH_SE_BUG=${WITH_SE_BUG:-0}
SCRUB_TARGETS=${SCRUB_TARGETS:-0}

if [ $WITH_SE_BUG -eq 1 -a $SCRUB_TARGETS -eq 1 ]
then
    DEFAULT_EXP_NAME=MultiLiTClsWithBugWithScrub
elif [ $WITH_SE_BUG -eq 1 ]
then
    DEFAULT_EXP_NAME=MultiLiTClsWithBug
elif [ $SCRUB_TARGETS -eq 1 ]
then
    DEFAULT_EXP_NAME=MultiLiTClsWithScrub
else
    DEFAULT_EXP_NAME=MultiLiTCls
fi


SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-$DEFAULT_EXP_NAME}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"

if [ $TARGET_FIT -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $WITH_SE_BUG -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.remove_se_hashtag false"
    fi
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    python -m mtse fit \
        -c configs/base/classic_target_classifier.yaml \
        $LOGGER_ARGS \
        --trainer.logger.version seed${seed}_target \
        --seed_everything $seed \
        $EXTRA_ARGS
else
    echo "Skipping target fitting"
fi

if [ $TARGET_TEST -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $WITH_SE_BUG -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.remove_se_hashtag false"
    fi
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    python -m mtse test \
        -c $LOGS_ROOT/seed${seed}_target/config.yaml \
        --trainer.logger.version seed${seed}_target_test \
        --ckpt_path $LOGS_ROOT/seed${seed}_target/checkpoints/*ckpt \
        $EXTRA_ARGS
else
    echo "Skipping target testing"
fi

if [ $TARGET_PRED -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $WITH_SE_BUG -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.remove_se_hashtag false"
    fi
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.scrub_targets true"
    fi

    version=seed${seed}_target_predict
    python -m mtse predict \
        -c $LOGS_ROOT/seed${seed}_target/config.yaml \
        --return_predictions false \
        --trainer.logger.version $version \
        --trainer.callbacks mtse.callbacks.TargetPredictionWriter \
        --trainer.callbacks.out_dir $LOGS_ROOT/$version \
        --trainer.callbacks.targets_path static/classic_merged_targets.txt \
        --trainer.callbacks.target_level mapped \
        --ckpt_path $LOGS_ROOT/seed${seed}_target/checkpoints/*ckpt \
        $EXTRA_ARGS
else
    echo "Skipping target prediction"
fi

if [ $STANCE_FIT -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $WITH_SE_BUG -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.target_train_corpus.transforms.remove_se_hashtag false"
    fi
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.target_train_corpus.transforms.scrub_targets true"
    fi

    python -m mtse fit \
        -c configs/base/classic_stance_classifier.yaml \
        $LOGGER_ARGS \
        --trainer.logger.version seed${seed}_stance \
        --seed_everything $seed \
        $EXTRA_ARGS
else
    echo "Skipping stance fitting"
fi

if [ $STANCE_TEST -eq 1 ]
then
    # We override the existing callback because we're not testing TSE this time
    train_dir=$LOGS_ROOT/seed${seed}_stance
    python -m mtse test \
        -c $train_dir/config.yaml \
        --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
        --trainer.logger.version seed${seed}_stance_test \
        --ckpt_path $train_dir/checkpoints/*ckpt
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
    train_dir=$LOGS_ROOT/seed${seed}_stance
    python -m mtse test \
        -c $train_dir/config.yaml \
        --ckpt_path $train_dir/checkpoints/*ckpt \
        --data.preds_dir $LOGS_ROOT/seed${seed}_target_predict \
        --data.target_input pred \
        --trainer.callbacks mtse.callbacks.TSEStatsCallback \
        --trainer.callbacks.full_metrics true \
        --trainer.logger.version seed${seed}_tse_test
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
    python -m mtse test \
        -c $LOGS_ROOT/seed${seed}_tse_test/config.yaml \
        --data.target_input label \
        --model.use_target_gt true \
        --trainer.logger.version seed${seed}_tse_test_gt
else
    echo "Skipping gt tse testing"
fi