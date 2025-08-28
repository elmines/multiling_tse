#!/bin/bash
ALL=${ALL:-0}
TARGET_FIT=${TARGET_FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
TARGET_PRED=${TARGET_PRED:-$ALL}
STANCE_FIT=${STANCE_FIT:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

SEEDS=${@:- 0 112 343}

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiLiTc}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

WITH_SE_BUG=${WITH_SE_BUG:-0}

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function v_target_train { echo LiTargetClassifier_seed${1}; }

function v_target_predict { echo $(v_target_train $1)_predict; }

function v_stance_train { echo LiStanceClassifier_seed${1}; }


if [ $TARGET_FIT -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $WITH_SE_BUG -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.transforms.remove_se_hashtag false"
    fi

    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/li_target_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version $(v_target_train $seed) \
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

    for seed in $SEEDS
    do
        python -m mtse test \
            -c $LOGS_ROOT/$(v_target_train $seed)/config.yaml \
            --data configs/data/li_tc_test.yaml \
            --trainer.logger.version $(v_target_train $seed)_test \
            --ckpt_path $LOGS_ROOT/$(v_target_train $seed)/checkpoints/*ckpt \
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
    for seed in $SEEDS
    do
        python -m mtse predict \
            -c $LOGS_ROOT/$(v_target_train $seed)/config.yaml \
            --data configs/data/li_tc_predict.yaml \
            --trainer.logger.version $(v_target_predict $seed) \
            --ckpt_path $LOGS_ROOT/$(v_target_train $seed)/checkpoints/*ckpt \
            $EXTRA_ARGS
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
            --data.val_corpus.target_preds_path $LOGS_ROOT/$(v_target_predict $seed)/target_preds.0.txt \
            --trainer.logger.version $(v_stance_train $seed) \
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
            -c $LOGS_ROOT/$(v_stance_train $seed)/config.yaml \
            --data configs/data/li_stance_test.yaml \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version $(v_stance_train $seed)_test \
            --ckpt_path $LOGS_ROOT/$(v_stance_train $seed)/checkpoints/*ckpt
    done
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_stance_train $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --data.corpora.target_preds_path $LOGS_ROOT/$(v_target_predict $seed)/target_preds.1.txt \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version LiTse_seed${seed}
    done
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        train_dir=$LOGS_ROOT/$(v_stance_train $seed)
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data configs/data/li_tse_test.yaml \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version LiGtTse_seed${seed} \
            --data.corpora.target_input label \
            --model.use_target_gt true
    done
else
    echo "Skipping gt tse testing"
fi