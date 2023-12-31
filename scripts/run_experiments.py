import argparse
from pathlib import Path
import pickle

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from openbustools import standardfeeds
from openbustools.traveltime import data_loader, model_utils


if __name__=="__main__":
    pl.seed_everything(42, workers=True)

    if torch.cuda.is_available():
        num_workers=4
        pin_memory=True
        accelerator="cuda"
    else:
        num_workers=0
        pin_memory=False
        accelerator="cpu"
    # num_workers=0
    # pin_memory=False
    # accelerator="cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', required=True)
    parser.add_argument('-mf', '--model_folder', required=True)
    parser.add_argument('-r', '--run_label', required=True)
    parser.add_argument('-trdf', '--train_data_folders', nargs='+', required=True)
    parser.add_argument('-tedf', '--test_data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--test_date', required=True)
    parser.add_argument('-tn', '--test_n', required=True)
    args = parser.parse_args()

    print("="*30)
    print(f"EXPERIMENTS")
    print(f"RUN: {args.run_label}")
    print(f"MODEL: {args.model_type}")
    print(f"TRAIN CITY DATA: {args.train_data_folders}")
    print(f"TEST CITY DATA: {args.test_data_folders}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    res = {}
    n_folds = 5
    test_dates = standardfeeds.get_date_list(args.test_date, int(args.test_n))
    for fold_num in range(n_folds):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = model_utils.load_model(args.model_folder, args.run_label, args.model_type, fold_num)
        res[fold_num] = {}

        print(f"EXPERIMENT: SAME CITY")
        test_data, holdout_routes, test_config = data_loader.load_h5(args.train_data_folders, test_dates, holdout_routes=model.holdout_routes, config=model.config)
        test_dataset = data_loader.H5Dataset(test_data)
        test_dataset.include_grid = model.include_grid
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
            logger=False,
            inference_mode=True
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        res[fold_num]['same_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: DIFFERENT CITY")
        test_data, holdout_routes, test_config = data_loader.load_h5(args.test_data_folders, test_dates, holdout_routes=model.holdout_routes, config=model.config)
        test_dataset = data_loader.H5Dataset(test_data)
        test_dataset.include_grid = model.include_grid
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
            logger=False,
            inference_mode=True
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        res[fold_num]['diff_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        test_data, holdout_routes, test_config = data_loader.load_h5(args.train_data_folders, test_dates, only_holdout=True, holdout_routes=model.holdout_routes, config=model.config)
        test_dataset = data_loader.H5Dataset(test_data)
        test_dataset.include_grid = model.include_grid
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
            logger=False,
            inference_mode=True
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        res[fold_num]['holdout'] = {'preds':preds, 'labels':labels}

    p = Path('.') / 'results' / args.run_label
    p.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f"./results/{args.run_label}/{args.model_type}.pkl", 'wb'))
    print(f"EXPERIMENTS COMPLETE")