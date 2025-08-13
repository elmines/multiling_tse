#!/bin/bash

for seed in 0 112 343
do
    target_exp_name=LiTargetClassifier_seed${seed}

    python -m mtse fit \
        -c configs/full/li_target_classifier.yaml \
        --trainer.logger.save_dir ./lightning_logs \
        --trainer.logger.name MultiLi \
        --trainer.logger.version $target_exp_name \
        --seed_everything $seed


    train_dir=lightning_logs/MultiLi/$target_exp_name
    ckpt_path=$(ls $train_dir/checkpoints/*ckpt)

    python -m mtse test \
        -c $train_dir/config.yaml \
        --data configs/data/li_tse_target_test_split.yaml \
        --trainer.logger.version ${target_exp_name}_test \
        --ckpt_path $ckpt_path

    python -m mtse predict \
        -c $train_dir/config.yaml \
        --data configs/data/li_tse_target_predict_all.yaml \
        --trainer.logger.version ${target_exp_name}_predict \
        --ckpt_path $ckpt_path

    continue

    ########## Stage 2: Stance training and TSE ##############

    tse_exp_name=LiTse_seed${seed}
    python -m mtse predict \
        -c configs/full/li_target_classifier.yaml \
        --trainer.logger.save_dir ./lightning_logs \
        --trainer.logger.name MultiLi \
        --trainer.logger.version $tse_exp_name \
        --seed_everything $seed


done