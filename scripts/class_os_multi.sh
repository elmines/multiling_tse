#!/bin/bash
ALL=${ALL:-0}
FIT=${FIT:-$ALL}

SEEDS=${@:- 0 112 343}

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiClassifierOneShot}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function v_target_train { echo ClassifierOneShot_seed${1}; }

if [ $FIT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/classifier_oneshot.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version $(v_target_train $seed) \
            --seed_everything $seed
    done
else
    echo "Skipping fitting"
fi
