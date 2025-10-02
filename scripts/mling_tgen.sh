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

MERGE_TARGETS=${MERGE_TARGETS:-0}
if [ $MERGE_TARGETS -eq 1 ]
then
    TARGETS_PATH=static/reduced_multiling_targets.txt
    MERGED_STEM="_merged"
else
    TARGETS_PATH=static/multiling_targets.txt
    MERGED_STEM=""
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
    version=fold${fold}_seed${seed}${MERGED_STEM}_target_map

    EXTRA_ARGS=""
    if [ $MERGE_TARGETS -eq 1 ]
    then
        EXTRA_ARGS="$EXTRA_ARGS --data.merge_independence true"
    fi

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

    $(dirname $0)/../utils/cat_preds.py $LOGS_ROOT $LOGS_ROOT/fold${fold}_seed${seed}${MERGED_STEM}_full_target_preds.csv $fold $seed
else
    echo "Skipping target mapping"
fi

if [ $TARGET_TEST -eq 1 ]
then
        EXTRA_ARGS=""
        if [ $MERGE_TARGETS -eq 1 ]
        then
            EXTRA_ARGS="$EXTRA_ARGS --data.merge_independence true"
        fi

        python -m mtse test \
            --model mtse.modules.PassthroughModule \
            --data mtse.data.TargetPredictionDataModule \
            --data.data_dir $LOGS_ROOT/fold${fold}_seed${seed}${MERGED_STEM}_target_map \
            --data.targets_path $TARGETS_PATH \
            --data.suffix_pattern .target_preds.csv \
            $EXTRA_ARGS \
            --trainer.logger lightning.pytorch.loggers.CSVLogger \
            $LOGGER_ARGS \
            --trainer.logger.version fold${fold}_seed${seed}${MERGED_STEM}_target_test \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes $((1 + $(wc -l < $TARGETS_PATH) ))
else
    echo "Skipping target testing"
fi

if [ $STANCE_FIT -eq 1 ]
then
    for seed in $SEEDS
    do
        python -m mtse fit \
            -c configs/base/m_stance_classifier.yaml \
            $LOGGER_ARGS \
            --trainer.logger.version seed${seed}_stance \
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
            -c $LOGS_ROOT/seed${seed}_stance/config.yaml \
            --data configs/data/m_stance_test.yaml \
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
        corpora_args=""
        dataloader_labels=""
        predict_dir=$LOGS_ROOT/seed${seed}_target_map
        prefix=""
        for target_pred_path in $predict_dir/target_preds.*.txt
        do
            data_part=$(basename $target_pred_path | cut -d. -f2)
            data_path=data/multiling/${data_part}_test.csv
            corpus_args="{class_path: mtse.data.StanceCorpus, init_args: {path: $data_path, target_preds_path: $target_pred_path}}"
            corpora_args="${corpora_args}${prefix}${corpus_args}"
            dataloader_labels="${dataloader_labels}${prefix}${data_part}"
            prefix=","
        done
        train_dir=$LOGS_ROOT/seed${seed}_stance
        python -m mtse test \
            -c $train_dir/config.yaml \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --data mtse.data.PredictDataModule \
            --data.corpora "[$corpora_args]" \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.dataloader_labels "[$dataloader_labels]" \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version seed${seed}_tse_test
    done
else
    echo "Skipping tse testing"
fi
