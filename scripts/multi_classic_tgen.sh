#!/bin/bash
ALL=${ALL:-0}
FT_EMBED=${FT_EMBED:-$ALL}
TARGET_FIT=${TARGET_FIT:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
TARGET_PRED=${TARGET_PRED:-$ALL}
STANCE_FIT=${STANCE_FIT:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

# Never run by default--something we only do for viz/debugging
AGG_PRED=${AGG_PRED:-0}

seed=${1:-0}

SCRUB_TARGETS=${SCRUB_TARGETS:-0}
if [ $SCRUB_TARGETS -eq 1 ]
then
    exp_suffix="_with_scrub"
fi

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MultiClassicTgen}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function embed_path { echo $LOGS_ROOT/ft_seed${seed}.model; }

if [ $TARGET_FIT -eq 1 ]
then
        python -m mtse fit \
            -c configs/base/classic_target_generator.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed}_target \
            --seed_everything $seed 
else
    echo "Skipping target fitting"
fi

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

if [ $TARGET_PRED -eq 1 ]
then
        scrub_targets=$( [ $SCRUB_TARGETS -eq 1 ] && echo "true" || echo "false" )

        version=seed${seed}_target_predict$exp_suffix
        python -m mtse predict \
            -c $LOGS_ROOT/seed${seed}_target/config.yaml \
            --return_predictions false \
            --model.predict_targets true \
            --data configs/data/classic_tse_test.yaml \
            --data.transforms "[{class_path: mtse.data.ClassicPreprocess, init_args: {scrub_targets: $scrub_targets}}]" \
            --trainer.logger.version $version \
            --trainer.callbacks mtse.callbacks.TargetPredictionWriter \
            --trainer.callbacks.out_dir $LOGS_ROOT/$version \
            --trainer.callbacks.embeddings_path $(embed_path $seed) \
            --trainer.callbacks.targets_path static/classic_merged_targets.txt \
            --ckpt_path $LOGS_ROOT/seed${seed}_target/checkpoints/*ckpt
else
    echo "Skipping target prediction"
fi

if [ $TARGET_TEST -eq 1 ]
then
        python -m mtse test \
            --model mtse.modules.TargetPredModule \
            --model.targets_path static/classic_merged_targets.txt \
            --model.map_file $LOGS_ROOT/seed${seed}_target_predict$exp_suffix/target_pred_map.json \
            --data configs/data/classic_tse_test.yaml \
            --trainer.logger lightning.pytorch.loggers.CSVLogger \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed}_target_test$exp_suffix \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes 19
else
    echo "Skipping target testing"
fi

# You won't see any more WITH_SE_BUG below--
# Li et al. only had that bug in the target prediction portion of the code

if [ $STANCE_FIT -eq 1 ]
then
    EXTRA_ARGS=""
    if [ $SCRUB_TARGETS -eq 1 ]
    then
        # Li et al. only did target scrubbing when predicting targets
        EXTRA_ARGS="$EXTRA_ARGS --data.target_train_corpus.transforms.scrub_targets true"
    fi

    python -m mtse fit \
        -c configs/base/classic_stance_classifier.yaml \
        $LOGGER_ARGS \
        --trainer.logger.version seed${seed}_stance$exp_suffix \
        --seed_everything $seed \
        $EXTRA_ARGS
else
    echo "Skipping stance fitting"
fi

if [ $STANCE_TEST -eq 1 ]
then
    train_dir=$LOGS_ROOT/seed${seed}_stance$exp_suffix
    python -m mtse test \
        -c $train_dir/config.yaml \
        --ckpt_path $train_dir/checkpoints/*ckpt \
        --data configs/data/classic_stance_test.yaml \
        --data.transforms '[mtse.data.ClassicPreprocess]' \
        --trainer.callbacks mtse.callbacks.StanceClassificationStatsCallback \
        --trainer.logger.version seed${seed}_stance_test$exp_suffix
    # TODO: Those --data lines do override the transform logic we had in config.yaml,
    # but in the future wanna make that more explicit here
else
    echo "Skipping stance testing"
fi

if [ $AGG_PRED -eq 1 ]
then
    version=seed${seed}_stance_predict${exp_suffix}
    train_dir=$LOGS_ROOT/seed${seed}_stance$exp_suffix
    stance_pred_dir=$LOGS_ROOT/$version
    python -m mtse predict \
        -c $train_dir/config.yaml \
        --ckpt_path $train_dir/checkpoints/*ckpt \
        --data configs/data/classic_tse_test.yaml \
        --data.transforms '[mtse.data.ClassicPreprocess]' \
        --trainer.callbacks mtse.callbacks.StancePredictionWriter \
        --trainer.callbacks.out_dir $stance_pred_dir \
        --trainer.logger.version $version

    out_dir=$LOGS_ROOT/seed${seed}_catpred${exp_suffix}
    mkdir -p $out_dir
    target_pred_dir=$LOGS_ROOT/seed${seed}_target_predict$exp_suffix
    for data_file in data/classic/*.test.csv
    do
        f_basename=$(basename $data_file)
        python -m mtse.agg_preds \
            -i $data_file \
            -o $out_dir/$f_basename.full.csv \
            --gen $target_pred_dir/$f_basename.target_gens.csv \
            --pred $target_pred_dir/$f_basename.target_preds.csv \
            --stance $stance_pred_dir/$f_basename.stance_preds.csv
    done
else
    echo "Skipping prediction CSV generation"
fi


function get_tse_transform_arg
{
    use_gt=$1
    target_map_path="$LOGS_ROOT/seed${seed}_target_predict$exp_suffix/target_pred_map.json"
    set_to_input=$( [ $use_gt -eq 1 ] && echo 'false' || echo 'true' )
    # Don't have to worry about target scrubbing here
    TRANSFORM_ARG="{class_path: mtse.data.SetTargetPred, init_args: {map_file: $target_map_path, set_to_input: $set_to_input}}"
    TRANSFORM_ARG="$TRANSFORM_ARG,{class_path: mtse.data.ClassicPreprocess}"
    TRANSFORM_ARG="[$TRANSFORM_ARG]"
    echo "$TRANSFORM_ARG"
}

if [ $TSE_TEST -eq 1 ]
then
    train_dir=$LOGS_ROOT/seed${seed}_stance$exp_suffix
    python -m mtse test \
        -c $train_dir/config.yaml \
        --ckpt_path $train_dir/checkpoints/*ckpt \
        --data configs/data/classic_tse_test.yaml \
        --data.transforms "$(get_tse_transform_arg 0)" \
        --trainer.callbacks mtse.callbacks.TSEStatsCallback \
        --trainer.callbacks.full_metrics true \
        --trainer.logger.version seed${seed}_tse_test$exp_suffix
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
    python -m mtse test \
        -c $LOGS_ROOT/seed${seed}_tse_test$exp_suffix/config.yaml \
        --data.transforms "$(get_tse_transform_arg 1)" \
        --model.use_target_gt true \
        --trainer.logger.version seed${seed}_tse_test_gt$exp_suffix
else
    echo "Skipping gt tse testing"
fi