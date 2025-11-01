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

set -x
SHORT_TARGETS=0
LLM_TARGETS=0
if [ -z $TARGET_TYPE ]
then
    TARGETS_PATH=static/multiling_targets.txt
    EXP_MOD=""
elif [ "$TARGET_TYPE" = llm ]
then
    TARGETS_PATH=static/llm_multiling_targets.txt
    EXP_MOD="_llm"
    LLM_TARGETS=1
elif [ "$TARGET_TYPE" = short ]
then
    TARGETS_PATH=static/short.txt
    EXP_MOD="_short"
    SHORT_TARGETS=1
else
    echo Invalid TARGET_TYPE=$TARGET_TYPE
    exit 1
fi

fold=${1:-0}
seed=${2:-0}

SAVE_DIR=${SAVE_DIR:-./lightning_logs}
EXP_NAME=${EXP_NAME:-MlingTgen}
LOGS_ROOT=$SAVE_DIR/$EXP_NAME

LOGGER_ARGS="--trainer.logger.save_dir $SAVE_DIR --trainer.logger.name $EXP_NAME"


function embed_path { echo $LOGS_ROOT/ft_seed${seed}.model; }


# Doesn't depend on fold
if [ $FT_EMBED -eq 1 ]
then
    mkdir -p $LOGS_ROOT
    python -m mtse.train_ft \
        --corpus_type standard \
        -i data/multiling/en_unrelated.csv \
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
        --data mtse.data.PattDataModule --data.test_corpora "[data/multiling/fold${fold}/*.test.csv]" \
        --model.predict_targets true \
        --trainer.logger.version $version \
        --trainer.callbacks+=mtse.callbacks.TargetPredictionWriter \
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
        if [ 1 -eq 1 || -e $out_dir ] && [ ! -z $(ls $out_dir) ]
        then
            echo Not overwriting existing $out_dir
            exit 1
        fi
        mkdir -p $out_dir

        in_files=()
        in_langs=()
        out_paths=()
        in_dir=$LOGS_ROOT/fold${fold}_seed${seed}_target_gen
        cp $in_dir/target_gen_map.json $out_dir/
        for target_path in $in_dir/*.test.csv.target_gens.csv
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
    version=fold${fold}_seed${seed}_target_map${EXP_MOD}

    EXTRA_ARGS=()
    if [ $LLM_TARGETS -eq 1 ]
    then
        EXTRA_ARGS+=(-c)
        EXTRA_ARGS+=("configs/llm_shorten_targets.yaml")
    elif [ $SHORT_TARGETS -eq 1 ]
    then
        EXTRA_ARGS+=(-c)
        EXTRA_ARGS+=("configs/shorten_targets.yaml")
    fi

    python -m mtse predict \
        --seed_everything $seed \
        --model mtse.modules.TargetPredModule \
        --model.targets_path $TARGETS_PATH \
        --model.map_file $LOGS_ROOT/fold${fold}_seed${seed}_target_translate/target_gen_map.json \
        --model.input_target_level generated \
        --model.with_lang true \
        --data mtse.data.PattDataModule --data.test_corpora "[data/multiling/fold${fold}/*.test.csv]" \
        --trainer.logger lightning.pytorch.loggers.CSVLogger \
        $LOGGER_ARGS \
        --trainer.logger.version $version \
        --trainer.callbacks mtse.callbacks.TargetPredictionWriter \
        --trainer.callbacks.out_dir $LOGS_ROOT/$version \
        --trainer.callbacks.targets_path $TARGETS_PATH \
        --trainer.callbacks.embeddings_path $(embed_path $seed) \
        --trainer.callbacks.target_level mapped \
        --trainer.callbacks.related_threshold 0.35 \
        "${EXTRA_ARGS[@]}"

    $(dirname $0)/../utils/cat_preds.py $LOGS_ROOT $LOGS_ROOT/fold${fold}_seed${seed}${EXP_MOD}_full_target_preds.csv $fold $seed
else
    echo "Skipping target mapping"
fi

if [ $TARGET_TEST -eq 1 ]
then
        EXTRA_ARGS=()
        if [ $LLM_TARGETS -eq 1 ]
        then
            EXTRA_ARGS+=(-c)
            EXTRA_ARGS+=("configs/llm_shorten_targets.yaml")
        elif [ $SHORT_TARGETS -eq 1 ]
        then
            EXTRA_ARGS+=(-c)
            EXTRA_ARGS+=("configs/shorten_targets.yaml")
        fi
        
        # Have to re-copy everything from MAP stage because of the return_predictions param won't let me reuse the YAML file...
        python -m mtse test \
            --seed_everything $seed \
            --model mtse.modules.TargetPredModule \
            --model.targets_path $TARGETS_PATH \
            --model.map_file $LOGS_ROOT/fold${fold}_seed${seed}_target_map${EXP_MOD}/target_pred_map.json \
            --model.input_target_level mapped \
            --model.with_lang true \
            --data mtse.data.PattDataModule --data.test_corpora "[data/multiling/fold${fold}/*.test.csv]" \
            --trainer.logger lightning.pytorch.loggers.CSVLogger \
            $LOGGER_ARGS \
            --trainer.logger.version fold${fold}_seed${seed}_target_test${EXP_MOD} \
            --trainer.callbacks mtse.callbacks.TargetClassificationStatsCallback \
            --trainer.callbacks.n_classes 10 \
            "${EXTRA_ARGS[@]}"
else
    echo "Skipping target testing"
fi

EXTRA_ARGS=()
if [ $LLM_TARGETS -eq 1 ]
then
    EXTRA_ARGS+=(-c "configs/llm_shorten_targets.yaml")
elif [ $SHORT_TARGETS -eq 1 ]
then
    EXTRA_ARGS+=(-c "configs/shorten_targets.yaml")
fi

if [ $STANCE_FIT -eq 1 ]
then

        python -m mtse fit \
            -c configs/base/m_stance_classifier.yaml \
            $LOGGER_ARGS \
            --model.targets_path $TARGETS_PATH \
            --data mtse.data.PattDataModule \
            --data.train_corpus "data/multiling/fold${fold}/*.train.csv" \
            --data.val_corpus "data/multiling/fold${fold}/*.val.csv" \
            --trainer.logger.version fold${fold}_seed${seed}_stance${EXP_MOD} \
            --seed_everything $seed \
            --trainer.max_epochs 1 \
            "${EXTRA_ARGS[@]}"
else
    echo "Skipping stance fitting"
fi

if [ $STANCE_TEST -eq 1 ]
then
        # Make each file a separate StanceCorpus object
        # StanceCorpus will give them better names that way
        corp_args=""
        add_comma=0
        for f in data/multiling/fold${fold}/*.test.csv
        do
            if [ $add_comma -eq 1 ]; then corp_args="$corp_args,"; fi
            corp_args="${corp_args}[$f]"
            add_comma=1
        done
        corp_args="[$corp_args]"

        train_version=fold${fold}_seed${seed}_stance${EXP_MOD}
        python -m mtse test \
            -c $LOGS_ROOT/$train_version/config.yaml \
            --data mtse.data.PattDataModule \
            --data.test_corpora "$corp_args" \
            $LOGGER_ARGS \
            --trainer.logger.version ${train_version}_test \
            --ckpt_path $LOGS_ROOT/$train_version/checkpoints/*ckpt \
            "${EXTRA_ARGS[@]}"
else
    echo "Skipping stance testing"
fi

if [ $TSE_TEST -eq 1 ]
then
        # Make each language a separate StanceCorpus object
        # StanceCorpus will give them better names that way
        corp_args=""
        add_comma=0
        for lang in ca es et fr it zh
        do
            if [ $add_comma -eq 1 ]; then corp_args="$corp_args,"; fi
            corp_args="${corp_args}{class_path: mtse.data.StanceCorpus,init_args: {name: $lang, patts: [data/multiling/fold${fold}/${lang}*.test.csv]}}"
            add_comma=1
        done
        corp_args="[$corp_args]"

        train_version=fold${fold}_seed${seed}_stance${EXP_MOD}
        python -m mtse test \
            -c $LOGS_ROOT/$train_version/config.yaml \
            --data mtse.data.PattDataModule \
            --data.test_corpora "$corp_args" \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            $LOGGER_ARGS \
            --trainer.logger.version fold${fold}_seed${seed}_tse_test${EXP_MOD} \
            --ckpt_path $LOGS_ROOT/$train_version/checkpoints/*ckpt \
            "${EXTRA_ARGS[@]}"
else
    echo "Skipping tse testing"
fi

if [ $GT_TSE_TEST -eq 1 ]
then
        train_dir=$LOGS_ROOT/fold${fold}_seed${seed}_stance${EXP_MOD}
        python -m mtse test \
            -c $train_dir/config.yaml \
            $LOGGER_ARGS \
            --ckpt_path $train_dir/checkpoints/*ckpt \
            --model.use_target_gt true \
            --data.preds_dir $LOGS_ROOT/fold${fold}_seed${seed}_target_map${EXP_MOD} \
            --data.target_input label \
            --trainer.callbacks mtse.callbacks.TSEStatsCallback \
            --trainer.callbacks.full_metrics true \
            --trainer.logger.version fold${fold}_seed${seed}_tse_test_gt${EXP_MOD} \
            "${EXTRA_ARGS[@]}"
else
    echo "Skipping gt tse testing"
fi
