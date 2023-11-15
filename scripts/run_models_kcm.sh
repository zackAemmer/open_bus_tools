#!/bin/bash
set -e

# Choose training and testing dates
TRAIN_DATE_START = 2023_03_15
TRAIN_NUM_DAYS = 7
TEST_DATE_START = 2023_03_22
TEST_NUM_DAYS = 7

# Choose training and testing bus networks
RUN_LABEL = kcm
TRAIN_NETWORK = './data/kcm_realtime/processed/'
TEST_NETWORK = './data/atb_realtime/processed/'

RUN_LABEL = atb
TRAIN_NETWORK = './data/atb_realtime/processed/'
TEST_NETWORK = './data/kcm_realtime/processed/'

RUN_LABEL = mix
TRAIN_NETWORK = './data/kcm_realtime/processed/ ./data/atb_realtime/processed/'
TEST_NETWORK = './data/rut_realtime/processed/'


cd ~/Desktop/open_bus_tools

# Train
python ./scripts/train_model.py -m FF -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m FF_STATIC -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m FF_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/train_model.py -m CONV -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m CONV_STATIC -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m CONV_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/train_model.py -m GRU -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m GRU_STATIC -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m GRU_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/train_model.py -m TRSF -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m TRSF_STATIC -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m TRSF_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/train_model.py -m DEEPTTE -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/train_model.py -m DEEPTTE_STATIC -mf ./logs/ -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

# Tune
python ./scripts/tune_model.py -m FF -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m FF_STATIC -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m FF_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/tune_model.py -m CONV -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m CONV_STATIC -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m CONV_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/tune_model.py -m GRU -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m GRU_STATIC -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m GRU_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/tune_model.py -m TRSF -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m TRSF_STATIC -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m TRSF_REALTIME -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

python ./scripts/tune_model.py -m DEEPTTE -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/tune_model.py -m DEEPTTE_STATIC -mf ./logs/ -r $RUN_LABEL -df $TEST_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS

# Experiment
python ./scripts/run_experiments.py -m FF -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m FF_STATIC -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m FF_REALTIME -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS

python ./scripts/run_experiments.py -m CONV -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m CONV_STATIC -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m CONV_REALTIME -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS

python ./scripts/run_experiments.py -m GRU -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m GRU_STATIC -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m GRU_REALTIME -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS

python ./scripts/run_experiments.py -m TRSF -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m TRSF_STATIC -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m TRSF_REALTIME -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS

python ./scripts/run_experiments.py -m DEEPTTE -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS
python ./scripts/run_experiments.py -m DEEPTTE_STATIC -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS

# Heuristic
python ./scripts/train_heuristics.py -r $RUN_LABEL -df $TRAIN_NETWORK -td $TRAIN_DATE_START -tn $TRAIN_NUM_DAYS
python ./scripts/run_heuristics.py -mf ./logs/ -r $RUN_LABEL -trdf $TRAIN_NETWORK -tedf $TEST_NETWORK -td $TEST_DATE_START -tn $TEST_NUM_DAYS