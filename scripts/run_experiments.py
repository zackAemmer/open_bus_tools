import json
import os
from pathlib import Path
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
from openbustools import data_utils


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    model_type = sys.argv[1]
    model_folder = sys.argv[2]
    network_name = sys.argv[3]
    train_city_data_folder = sys.argv[4]
    test_city_data_folder = sys.argv[5]
    train_date = sys.argv[6]
    train_n = sys.argv[7]
    test_date = sys.argv[8]
    test_n = sys.argv[9]
    train_dates = data_utils.get_date_list(train_date, int(train_n))
    test_dates = data_utils.get_date_list(test_date, int(test_n))

    # grid_s_size=500
    # if network_folder=="kcm/":
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # elif network_folder=="atb/":
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    # else:
    #     holdout_routes=None

    if torch.cuda.is_available():
        num_workers=4
        pin_memory=True
        accelerator="auto"
    else:
        num_workers=4
        pin_memory=False
        accelerator="cpu"

    print("="*30)
    print(f"EXPERIMENTS")
    print(f"TRAIN CITY DATA: {train_city_data_folder}")
    print(f"TEST CITY DATA: {test_city_data_folder}")
    print(f"MODEL: {model_type}")
    print(f"NETWORK: {network_name}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    # print(f"Building grid on fold testing data")
    # test_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/test", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    # test_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    # test_ngrid.add_grid_content(test_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    # test_ngrid.build_cell_lookup()
    # test_dataset.grid = test_ngrid

    res = {}
    n_folds = 5
    for fold_num in range(n_folds):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = model_utils.load_model(model_folder, network_name, model_type, fold_num)
        res[fold_num] = {}

        print(f"EXPERIMENT: SAME CITY")
        test_dataset = data_loader.ContentDataset(train_city_data_folder, test_dates, holdout_type='specify', holdout_routes=model.holdout_routes)
        test_dataset.config = model.config
        test_loader = DataLoader(
            test_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            logger=False
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = torch.concat([x['preds'] for x in preds_and_labels]).numpy()
        labels = torch.concat([x['labels'] for x in preds_and_labels]).numpy()
        preds = data_loader.denormalize(preds, model.config['cumul_time_s'])
        labels = data_loader.denormalize(labels, model.config['cumul_time_s'])
        res[fold_num]['same_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: DIFFERENT CITY")
        test_dataset = data_loader.ContentDataset(test_city_data_folder, test_dates)
        test_dataset.config = model.config
        test_loader = DataLoader(
            test_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            logger=False
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = torch.concat([x['preds'] for x in preds_and_labels]).numpy()
        labels = torch.concat([x['labels'] for x in preds_and_labels]).numpy()
        preds = data_loader.denormalize(preds, model.config['cumul_time_s'])
        labels = data_loader.denormalize(labels, model.config['cumul_time_s'])
        res[fold_num]['diff_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        test_dataset = data_loader.ContentDataset(train_city_data_folder, test_dates, holdout_type='specify', only_holdout=True, holdout_routes=model.holdout_routes)
        test_dataset.config = model.config
        test_loader = DataLoader(
            test_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            logger=False
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = torch.concat([x['preds'] for x in preds_and_labels]).numpy()
        labels = torch.concat([x['labels'] for x in preds_and_labels]).numpy()
        preds = data_loader.denormalize(preds, model.config['cumul_time_s'])
        labels = data_loader.denormalize(labels, model.config['cumul_time_s'])
        res[fold_num]['holdout'] = {'preds':preds, 'labels':labels}

    p = Path('.') / 'results' / network_name
    p.mkdir(exist_ok=True)
    pickle.dump(res, open(f"./results/{network_name}/{model_type}.pkl", "wb"))
    print(f"EXPERIMENTS COMPLETE")