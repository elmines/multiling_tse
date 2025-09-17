#!/bin/bash
ALL=${ALL:-0}
TARGET_FIT=${TARGET_FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
TARGET_PRED=${TARGET_PRED:-$ALL}
STANCE_FIT=${STANCE_FIT:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

SEEDS=${@:- 0 1 2}

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

    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/li_target_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed}_target \
            --seed_everything $seed \
            $EXTRA_ARGS
    done
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

    for seed in $SEEDS
    do
        python -m mtse test \
            -c $LOGS_ROOT/seed${seed}_target/config.yaml \
            --data configs/data/li_tc_test.yaml \
            --trainer.logger.version seed${seed}_target_test \
            --ckpt_path $LOGS_ROOT/seed${seed}_target/checkpoints/*ckpt \
            $EXTRA_ARGS
    done
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

    for seed in $SEEDS
    do
        version=seed${seed}_target_predict
        python -m mtse predict \
            -c $LOGS_ROOT/seed${seed}_target/config.yaml \
            --return_predictions false \
            --data configs/data/li_tc_predict.yaml \
            --trainer.logger.version $version \
            --trainer.callbacks mtse.callbacks.TargetPredictionWriter \
            --trainer.callbacks.out_dir $LOGS_ROOT/$version \
            --trainer.callbacks.targets_path static/li_merged_targets.txt \
            --ckpt_path $LOGS_ROOT/seed${seed}_target/checkpoints/*ckpt \
            $EXTRA_ARGS
    done
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

    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/li_stance_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed}_stance \
            --seed_everything $seed \
            $EXTRA_ARGS
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
            -c $LOGS_ROOT/seed${seed}_stance/config.yaml \
            --data configs/data/li_stance_test.yaml \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version seed${seed}_stance_test \
            --ckpt_path $LOGS_ROOT/seed${seed}_stance/checkpoints/*ckpt
    done
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/seed${seed}_stance
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --data.corpora.target_preds_path $LOGS_ROOT/seed${seed}_target_predict/target_preds.1.txt \
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
        train_dir=$LOGS_ROOT/seed${seed}_stance
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