#!/bin/bash
ALL=${ALL:-0}
FT_EMBED=${FT_EMBED:-$ALL}
TARGET_FIT=${TARGET_FIT:-$ALL}
TARGET_GEN=${TARGET_GEN:-$ALL}
TARGET_TRANS=${TARGET_TRANS:-$ALL}
TARGET_MAP=${TARGET_MAP:-$ALL}
TARGET_TEST=${TARGET_TEST:-$ALL}
STANCE_FIT=${STANCE_FIT:-$ALL}
STANCE_TEST=${STANCE_TEST:-$ALL}
TSE_TEST=${TSE_TEST:-$ALL}
GT_TSE_TEST=${GT_TSE_TEST:-$ALL}

SHORT_TARGETS=${SHORT_TARGETS:-0}
if [ $SHORT_TARGETS -eq 1 ]
then
    TARGETS_PATH=static/shortened_multiling_targets.txt
    EXP_MOD="_short"
else
    TARGETS_PATH=static/multiling_targets.txt
    EXP_MOD=""
fi

fold=${1:-0}
seed=${2:-0}


SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MlingTGen}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function embed_path { echo $LOGS_ROOT/ft_seed${seed}.model; }

function extract_langs
{
    for in_path in $@
    do
        echo $in_path | sed -E 's/.*target_gens\.([a-z]{2})_.*/\1/'
    done
}

# Doesn't depend on fold
if [ $FT_EMBED -eq 1 ]
then
    mkdir -p $LOGS_ROOT
    python -m mtse.train_ft \
        --corpus_type standard \
        -i data/multiling/en_unrelated_all.csv \
        --seed $seed \
        --embed 256 \
        -o $(embed_path $seed) \
        --epochs 500 
else
    echo "Skipping FastText embedding"
fi

# Doesn't depend on fold
if [ $TARGET_FIT -eq 1 ]
then
    python -m mtse fit \
        -c configs/base/mt5_target_generator.yaml \
        $LOGGER_ARGS \
        --trainer.logger.version seed${seed}_target \
        --seed_everything $seed 
else
    echo "Skipping target fitting"
fi

if [ $TARGET_GEN -eq 1 ]
then
    version=fold${fold}_seed${seed}_target_gen

    python -m mtse predict \
        -c $LOGS_ROOT/seed${seed}_target/config.yaml \
        $LOGGER_ARGS \
        --return_predictions false \
        --data mtse.data.DirDataModule \
        --model.predict_targets true \
        --data.data_dir data/multiling/fold${fold} \
        --trainer.logger.version $version \
        --trainer.callbacks mtse.callbacks.TargetPredictionWriter \
        --trainer.callbacks.targets_path $TARGETS_PATH \
        --trainer.callbacks.target_level generated \
        --trainer.callbacks.out_dir $LOGS_ROOT/$version \
        --trainer.callbacks.embeddings_path $(embed_path $seed) \
        --ckpt_path $LOGS_ROOT/seed${seed}_target/checkpoints/*ckpt
else
    echo "Skipping target generation"
fi

if [ $TARGET_TRANS -eq 1 ]
then
        out_dir=$LOGS_ROOT/fold${fold}_seed${seed}_target_translate
        if [ -e $out_dir -a ! -z $(ls $out_dir) ]
        then
            echo Not overwriting existing $out_dir
            exit 1
        fi
        mkdir -p $out_dir

        in_files=()
        in_langs=()
        out_paths=()
        for target_path in $LOGS_ROOT/fold${fold}_seed${seed}_target_gen/*target_gens.csv
        do
            in_files+=($target_path)
            in_langs+=($(basename $target_path | cut -c1-2))
            out_paths+=($out_dir/$(basename $target_path))
        done

        python -m mtse.translate pred \
            -i ${in_files[@]} \
            --lang ${in_langs[@]} \
            -o ${out_paths[@]}
else
    echo "Skipping target translation"
fi

if [ $TARGET_MAP -eq 1 ]
then
    version=fold${fold}_seed${seed}${EXP_MOD}_target_map

    EXTRA_ARGS=""
    if [ $SHORT_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.shorten_targets true"
    fi
    TARGETS_PATH=static/shortened_multiling_targets.txt
    python -m mtse predict \
        --seed_everything $seed \
        --model mtse.modules.PassthroughModule \
        --data mtse.data.TargetPredictionDataModule \
        --data.data_dir $LOGS_ROOT/fold${fold}_seed${seed}_target_translate \
        --data.targets_path $TARGETS_PATH \
        --data.suffix_pattern .target_gens.csv \
        --data.with_generated true \
        --data.with_untranslated true \
        $EXTRA_ARGS \
        --trainer.logger lightning.pytorch.loggers.CSVLogger \
        $LOGGER_ARGS \
        --trainer.logger.version $version \
        --trainer.callbacks mtse.callbacks.TargetPredictionWriter \
        --trainer.callbacks.out_dir $LOGS_ROOT/$version \
        --trainer.callbacks.targets_path $TARGETS_PATH \
        --trainer.callbacks.embeddings_path $(embed_path $seed) \
        --trainer.callbacks.target_level mapped \
        --trainer.callbacks.related_threshold 0.35

    $(dirname $0)/../utils/cat_preds.py $LOGS_ROOT $LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_full_target_preds.csv $fold $seed
else
    echo "Skipping target mapping"
fi

if [ $TARGET_TEST -eq 1 ]
then
        EXTRA_ARGS=""
        if [ $SHORT_TARGETS -eq 1 ]
        then
            EXTRA_ARGS="$EXTRA_ARGS --data.shorten_targets true"
        fi

        python -m mtse test \
            --model mtse.modules.PassthroughModule \
            --data mtse.data.TargetPredictionDataModule \
            --data.data_dir $LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_target_map \
            --data.targets_path $TARGETS_PATH \
            --data.suffix_pattern .target_preds.csv \
            $EXTRA_ARGS \
            --trainer.logger lightning.pytorch.loggers.CSVLogger \
            $LOGGER_ARGS \
            --trainer.logger.version fold${fold}_seed${seed}${EXP_MOD}_target_test \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes $((1 + $(wc -l < $TARGETS_PATH) ))
else
    echo "Skipping target testing"
fi

if [ $STANCE_FIT -eq 1 ]
then
        EXTRA_ARGS=()
        if [ $SHORT_TARGETS -eq 1 ]
        then
            EXTRA_ARGS+=(--data.transforms)
            EXTRA_ARGS+=("[{class_path : mtse.data.MergeIndependence}]")
        fi

        python -m mtse fit \
            -c configs/base/m_stance_classifier.yaml \
            $LOGGER_ARGS \
            --model.targets_path $TARGETS_PATH \
            --data mtse.data.DirDataModule \
            --data.data_dir data/multiling/fold${fold} \
            --trainer.logger.version fold${fold}_seed${seed}${EXP_MOD}_stance \
            --seed_everything $seed \
            "${EXTRA_ARGS[@]}"
else
    echo "Skipping stance fitting"
fi

if [ $STANCE_TEST -eq 1 ]
then
        train_dir=fold${fold}_seed${seed}${EXP_MOD}_stance
        # We override the existing callback because we're not testing TSE this time
        python -m mtse test \
            -c $LOGS_ROOT/$train_dir/config.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version ${train_dir}_test \
            --ckpt_path $LOGS_ROOT/$train_dir/checkpoints/*ckpt
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
        train_dir=$LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_stance
        python -m mtse test \
            -c $train_dir/config.yaml \
            $LOGGER_ARGS \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data.preds_dir $LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_target_map \
            --data.target_input pred \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version fold${fold}_seed${seed}${EXP_MOD}_tse_test
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
        train_dir=$LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_stance
        python -m mtse test \
            -c $train_dir/config.yaml \
            $LOGGER_ARGS \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --model.use_target_gt true \
            --data.preds_dir $LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_target_map \
            --data.target_input label \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version fold${fold}_seed${seed}${EXP_MOD}_tse_test_gt
else
    echo "Skipping gt tse testing"
fi
