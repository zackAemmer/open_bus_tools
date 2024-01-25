import argparse
import logging
from pathlib import Path
import pickle

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.model_selection import KFold

from openbustools import standardfeeds
from openbustools.traveltime import data_loader
from openbustools.traveltime.models import avg_speed


if __name__=="__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('train_heuristics')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_label', required=True)
    parser.add_argument('-df', '--data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--train_date', required=True)
    parser.add_argument('-tn', '--train_n', required=True)
    args = parser.parse_args()

    logger.info(f"RUN: {args.run_label}")
    logger.info(f"MODEL: HEURISTICS")
    logger.info(f"DATA: {args.data_folders}")
    logger.info(f"START: {args.train_date}")
    logger.info(f"DAYS: {args.train_n}")

    k_fold = KFold(2, shuffle=True, random_state=42)
    train_days = standardfeeds.get_date_list(args.train_date, int(args.train_n))
    train_days = [x.split(".")[0] for x in train_days]
    train_dataset = data_loader.NumpyDataset(
        args.data_folders,
        train_days,
        holdout_routes=data_loader.HOLDOUT_ROUTES,
        load_in_memory=False
    )

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(len(train_dataset)))):
        logger.info(f"FOLD: {fold_num}")
        model = avg_speed.AvgSpeedModel('AVG', train_dataset, train_dataset.config, train_dataset.holdout_routes, idx=train_idx)
        save_path = Path("logs", f"{args.run_label}", f"AVG-{fold_num}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(save_path, 'wb'))
    logger.info(f"{model.model_name} TRAINING COMPLETE")