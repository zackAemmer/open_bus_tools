import argparse
import pickle

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from openbustools import standardfeeds
from openbustools.traveltime import data_loader
from openbustools.traveltime.models import avg_speed


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_label', required=True)
    parser.add_argument('-df', '--data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--train_date', required=True)
    parser.add_argument('-tn', '--train_n', required=True)
    args = parser.parse_args()

    train_dates = standardfeeds.get_date_list(args.train_date, int(args.train_n))

    print("="*30)
    print(f"TRAINING")
    print(f"RUN: {args.run_label}")
    print(f"MODEL: HEURISTICS")
    print(f"DATA: {args.data_folders}")

    k_fold = KFold(5, shuffle=True, random_state=42)
    train_data, holdout_routes, train_config = data_loader.load_h5(args.data_folders, train_dates, holdout_routes=data_loader.HOLDOUT_ROUTES)
    train_dataset = data_loader.H5Dataset(train_data)

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(train_dataset.__len__()))):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = avg_speed.AvgSpeedModel('AVG', train_dataset, idx=train_idx)
        model.config = train_config
        model.holdout_routes = holdout_routes
        pickle.dump(model, open(f"./logs/{args.run_label}/AVG_{fold_num}.pkl", 'wb'))
    print(f"TRAINING COMPLETE")