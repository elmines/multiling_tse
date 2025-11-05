#!/bin/bash
ALL=${ALL:-0}

FT_EMBED=${FT_EMBED:-$ALL}
FIT=${FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

seed=${1:-0}

SCRUB_TARGETS=${SCRUB_TARGETS:-0}
WITH_SE_BUG=${WITH_SE_BUG:-0}
if [ $SCRUB_TARGETS -eq 1 ]
then
    exp_suffix="_with_scrub"
fi

if [ $WITH_SE_BUG -eq 1 -a $SCRUB_TARGETS -eq 1 ]
then
    exp_suffix="_with_bug_with_scrub"
elif [ $WITH_SE_BUG -eq 1 ]
then
    exp_suffix="_with_bug"
elif [ $SCRUB_TARGETS -eq 1 ]
then
    exp_suffix="_with_scrub"
else
    exp_suffix=""
fi


SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiOneshotTgen}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function embed_path { echo $LOGS_ROOT/ft_seed${seed}.model; }

if [ $FT_EMBED -eq 1 ]
then
    mkdir -p $LOGS_ROOT
    in_files=(data/classic/*.train.csv)
    corpus_types=$(for f in ${in_files[@]}; do echo " standard"; done)
        python -m mtse.train_ft \
            --corpus_type $corpus_types \
            -i ${in_files[@]} \
            --seed $seed \
            --embed 256 \
            -o $(embed_path $seed) \
            --epochs 500 
else
    echo "Skipping FastText embedding"
fi

if [ $FIT -eq 1 ]
then
        python -m mtse fit \
            -c configs/base/oneshot_tgen.yaml \
            --model.embeddings_path $(embed_path $seed) \
            $( [ $SCRUB_TARGETS -eq 1 ] && echo --data.stance_train_corpus.transforms.scrub_targets     'true' --data.stance_val_corpus.transforms.scrub_targets     'true' ) \
            $( [ $WITH_SE_BUG   -eq 1 ] && echo --data.stance_train_corpus.transforms.remove_se_hashtag 'true' --data.stance_val_corpus.transforms.remove_se_hashtag 'true') \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed}${exp_suffix} \
            --seed_everything $seed
else
    echo "Skipping fitting"
fi

train_dir=$LOGS_ROOT/seed${seed}${exp_suffix}

TRANSFORM_ARGS=(--data.transforms "[{class_path: mtse.data.ClassicPreprocess}]")
if [ $SCRUB_TARGETS -eq 1 ]
then
    TRANSFORM_ARGS+=(--data.transforms.scrub_targets 'true')
fi
if [ $WITH_SE_BUG -eq 1 ]
then
    TRANSFORM_ARGS+=(--data.transforms.remove_se_hashtag 'false')
fi

if [ $TARGET_TEST -eq 1 ]
then
        python -m mtse test \
            -c $train_dir/config.yaml \
            --data configs/data/classic_tse_test.yaml \
            "${TRANSFORM_ARGS[@]}" \
            --trainer.logger.version seed${seed}_target_test${exp_suffix} \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes 19 \
            --ckpt_path $train_dir/checkpoints/*ckpt
else
    echo "Skipping target testing"
fi

if [ $STANCE_TEST -eq 1 ]
then
        # We override the existing callback because we're not testing TSE this time
        python -m mtse test \
            -c $train_dir/config.yaml \
            --data configs/data/classic_stance_test.yaml \
            $( [ $WITH_SE_BUG   -eq 1 ] && echo --data.transforms.remove_se_hashtag 'true') \
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version seed${seed}_stance_test${exp_suffix} \
            --ckpt_path $train_dir/checkpoints/*ckpt 
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
        python -m mtse test \
            -c $train_dir/config.yaml \
            --data configs/data/classic_tse_test.yaml \
            "${TRANSFORM_ARGS[@]}" \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --trainer.logger.version seed${seed}_tse_test${exp_suffix}
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
        python -m mtse test \
            -c $train_dir/config.yaml \
            --data configs/data/classic_tse_test.yaml \
            "${TRANSFORM_ARGS[@]}" \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --trainer.logger.version seed${seed}_tse_test_gt${exp_suffix} \
            --model.use_target_gt true 
else
    echo "Skipping gt tse testing"
fi