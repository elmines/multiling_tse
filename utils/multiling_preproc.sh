#!/bin/bash

SCRIPT_DIR=$(dirname $0)/corpus_scripts
DATA_DIR=$(dirname $0)/../data

$SCRIPT_DIR/preproc_globalvoices.py $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_cstance_data.py $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_sardi_data.py   $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_cs_data.py      $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_nlpcc_data.py   $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_cic_data.py     $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_hi_data.py      $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_et_data.py      $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/preproc_lai_data.py     $DATA_DIR/multiling/raw $DATA_DIR/multiling/
$SCRIPT_DIR/part_kptimes.py