#!/bin/bash
set -e

RUN_NAME="big_run"
MODEL_NAME="FF"

cd ~/Desktop/open_bus_tools

# python ./scripts/train_model.py "FF" "./logs/" "kcm" "./data/kcm_realtime/processed/" "2023_03_15" "3"
python ./scripts/tune_model.py "FF" "./logs/" "kcm" "./data/atb_realtime/processed/" "2023_03_15" "3"
python ./scripts/fit_heuristics.py "kcm" "./data/atb_realtime/processed/" "2023_03_15" "3"
python ./scripts/run_experiments.py "FF" "./logs/" "kcm" "./data/kcm_realtime/processed/" "./data/atb_realtime/processed/" "2023_03_22" "3"