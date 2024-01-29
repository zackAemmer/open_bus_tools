import argparse
import logging
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
    logger = logging.getLogger('run_experiments')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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

    logger.info(f"RUN: {args.run_label}")
    logger.info(f"MODEL: {args.model_type}")
    logger.info(f"TRAIN CITY DATA: {args.train_data_folders}")
    logger.info(f"TEST CITY DATA: {args.test_data_folders}")
    logger.info(f"START: {args.test_date}")
    logger.info(f"DAYS: {args.test_n}")
    logger.info(f"num_workers: {num_workers}")
    logger.info(f"pin_memory: {pin_memory}")

    res = {}
    n_folds = 2
    test_days = standardfeeds.get_date_list(args.test_date, int(args.test_n))
    test_days = [x.split(".")[0] for x in test_days]
    for fold_num in range(n_folds):
        logger.info(f"MODEL {args.model_type}, FOLD: {fold_num}")
        model = model_utils.load_model(args.model_folder, args.run_label, args.model_type, fold_num)
        res[fold_num] = {}

        logger.info(f"EXPERIMENT: SAME CITY")
        test_dataset = data_loader.NumpyDataset(
            args.train_data_folders,
            test_days,
            holdout_routes=model.holdout_routes,
            load_in_memory=False,
            include_grid=True if "REALTIME" in args.model_type.split("_") else False,
            config=model.config
        )
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

        logger.info(f"EXPERIMENT: DIFFERENT CITY")
        test_dataset = data_loader.NumpyDataset(
            args.test_data_folders,
            test_days,
            holdout_routes=model.holdout_routes,
            load_in_memory=False,
            include_grid=True if "REALTIME" in args.model_type.split("_") else False,
            config=model.config
        )
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

        logger.info(f"EXPERIMENT: HOLDOUT ROUTES")
        test_dataset = data_loader.NumpyDataset(
            args.train_data_folders,
            test_days,
            holdout_routes=model.holdout_routes,
            load_in_memory=False,
            include_grid=True if "REALTIME" in args.model_type.split("_") else False,
            config=model.config,
            only_holdouts=True
        )
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

    p = Path("results") / args.run_label
    p.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(p / f"{args.model_type}.pkl", 'wb'))
    logger.info(f"{model.model_name} EXPERIMENTS COMPLETE")