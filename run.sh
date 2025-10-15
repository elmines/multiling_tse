#!/bin/bash



for fold in {0..4}
do
    EXP_NAME=Short TARGET_TYPE=short SAVE_DIR=/blue/bonniejdorr/ethanlmines/9oct_lit/ TARGET_TEST=1 scripts/mling_tgen.sh $fold 0
done