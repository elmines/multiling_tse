#!/bin/bash
ALL=${ALL:-0}

FT_EMBED=${FT_EMBED:-$ALL}
FIT=${FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

# Never run by default--something we only do for viz/debugging
PRED=${PRED:-0}

seed=${1:-0}

SCRUB_TARGETS=${SCRUB_TARGETS:-0}
if [ $SCRUB_TARGETS -eq 1 ]
then
    exp_suffix="_with_scrub"
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
            $( [ $SCRUB_TARGETS -eq 1 ] && echo --data.stance_val_corpus.transforms.scrub_targets 'true' ) \
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
            --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
            --trainer.logger.version seed${seed}_stance_test${exp_suffix} \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data.transforms "[{class_path: mtse.data.ClassicPreprocess, init_args: { scrub_targets: false , remove_se_hashtag: true } }]"
            # No target scrubbing when doing pure stance testing
            # And since the TSE authors scrubbed the hashtag in stance testing, so will we here
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

if [ $PRED -eq 1 ]
then
        version=seed${seed}_pred${exp_suffix}
        out_dir=$LOGS_ROOT/$version
        python -m mtse predict \
            -c $train_dir/config.yaml \
            --data configs/data/classic_tse_test.yaml \
            "${TRANSFORM_ARGS[@]}" \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --return_prediction false \
            --model.map_targets false \
            --trainer.callbacks "[]" \
            --trainer.callbacks+=mtse.callbacks.StancePredictionWriter \
            --trainer.callbacks.out_dir $out_dir \
            --trainer.callbacks+=mtse.callbacks.TargetPredictionWriter \
            --trainer.callbacks.out_dir $out_dir \
            --trainer.callbacks.embeddings_path $(embed_path $seed) \
            --trainer.callbacks.targets_path static/classic_merged_targets.txt \
            --trainer.logger.version $version
else
    echo "Skipping tse testing"
fi
