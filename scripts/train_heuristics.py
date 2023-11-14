import argparse
import json
import os
import pickle
import shutil
import time

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import KFold
from tabulate import tabulate
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler

from openbustools import standardfeeds
from openbustools.traveltime import data_loader, grids, model_utils
from openbustools.traveltime.models import avg_speed


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network_name', required=True)
    parser.add_argument('-df', '--data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--train_date', required=True)
    parser.add_argument('-tn', '--train_n', required=True)
    args = parser.parse_args()

    train_dates = standardfeeds.get_date_list(args.train_date, int(args.train_n))

    print("="*30)
    print(f"TRAINING")
    print(f"DATA: '{args.data_folders}'")
    print(f"MODEL: HEURISTICS")

    k_fold = KFold(5, shuffle=True, random_state=42)
    train_dataset = data_loader.ContentDataset(args.data_folders, train_dates, holdout_type='create')

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(train_dataset.__len__()))):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = avg_speed.AvgSpeedModel('AVG', train_dataset.data.loc[train_idx])
        model.config = train_dataset.config
        model.holdout_routes = train_dataset.holdout_routes
        pickle.dump(model, open(f"./logs/{args.network_name}/AVG_{fold_num}.pkl", 'wb'))
    print(f"TRAINING COMPLETE")