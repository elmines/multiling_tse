#!/bin/bash

TARGET_FIT=${TARGET_FIT:-1}
TARGET_TEST=${TARGET_TEST:-1}
TARGET_PRED=${TARGET_PRED:-1}
STANCE_FIT=${STANCE_FIT:-1}
SEEDS=${@:- 0 112 343}

SAVE_DIR=./lightning_logs
EXP_NAME=MultiLi
LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function get_target_train_version
{
    seed=$1
    echo LiTargetClassifier_seed${seed}
}
function get_target_predict_version
{
    seed=$1
    echo $(get_target_train_version $seed)_predict
}

function get_target_train_dir
{
    seed=$1
    echo $SAVE_DIR/$EXP_NAME/$(get_target_train_version $seed)
}
function get_target_preds_dir
{
    seed=$1
    echo $SAVE_DIR/$EXP_NAME/$(get_target_predict_version $seed)
}

function get_stance_exp_name
{
    seed=$1
    echo LiStanceClassifier_seed${seed}
}

function get_stance_train_dir
{
    seed=$1
    echo $SAVE_DIR/$EXP_NAME/$(get_stance_exp_name $seed)
}

if [ $TARGET_FIT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/full/li_target_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version $(get_target_train_version $seed) \
            --seed_everything $seed
    done
else
    echo "Skipping target fitting"
fi

if [ $TARGET_TEST -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse test \
            -c $(get_target_train_dir $seed)/config.yaml \
            --data configs/data/li_tse_target_test_split.yaml \
            --trainer.logger.version $(get_target_train_version $seed)_test \
            --ckpt_path $(ls $(get_target_train_dir $seed)/checkpoints/*ckpt)
    done
else
    echo "Skipping target testing"
fi

if [ $TARGET_PRED -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse predict \
            -c $(get_target_train_dir $seed)/config.yaml \
            --data configs/data/li_tse_target_predict_all.yaml \
            --trainer.logger.version $(get_target_predict_version $seed) \
            --ckpt_path $(ls $(get_target_train_dir $seed)/checkpoints/*ckpt)
    done
else
    echo "Skipping target prediction"
fi

if [ $STANCE_FIT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/full/li_stance_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version $(get_stance_exp_name $seed) \
            --seed_everything $seed
    done
else
    echo "Skipping stance fitting"
fi