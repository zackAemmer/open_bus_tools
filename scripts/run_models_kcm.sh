#!/bin/bash
set -e


cd ~/Desktop/open_bus_tools

python ./scripts/train_model.py "GRU" "./logs/" "kcm" "./data/kcm_realtime/processed/" "2023_03_15" "7"
python ./scripts/tune_model.py "GRU" "./logs/" "kcm" "./data/atb_realtime/processed/" "2023_03_15" "7"
python ./scripts/run_experiments.py "GRU" "./logs/" "kcm" "./data/kcm_realtime/processed/" "./data/atb_realtime/processed/" "2023_03_22" "7"