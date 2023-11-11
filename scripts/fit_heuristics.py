import json
import os
import pickle
import shutil
import sys
import time

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn import metrics
from sklearn.model_selection import KFold
from tabulate import tabulate
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler

from openbustools.traveltime import data_loader, grids, model_utils
from openbustools.traveltime.models import avg_speed
from openbustools import data_utils


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    network_name = sys.argv[1]
    data_folder = sys.argv[2]
    train_date = sys.argv[3]
    train_n = sys.argv[4]
    train_dates = data_utils.get_date_list(train_date, int(train_n))

    print("="*30)
    print(f"TRAINING")
    print(f"DATA: '{data_folder}'")
    print(f"MODEL: HEURISTICS")

    k_fold = KFold(5, shuffle=True, random_state=42)
    train_dataset = data_loader.ContentDataset(data_folder, train_dates, holdout_type='create')

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(train_dataset.__len__()))):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = avg_speed.AvgSpeedModel('AVG', train_dataset.data.loc[train_idx])
        model.config = train_dataset.config
        model.holdout_routes = train_dataset.holdout_routes
        pickle.dump(model, open(f"./logs/{network_name}/AVG_{fold_num}.pkl", 'wb'))
    print(f"TRAINING COMPLETE")