import argparse
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from openbustools import standardfeeds
from openbustools.traveltime import data_loader, model_utils


if __name__=="__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('tune_model')
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
    parser.add_argument('-df', '--data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--train_date', required=True)
    parser.add_argument('-tn', '--train_n', required=True)
    args = parser.parse_args()

    logger.info(f"RUN: {args.run_label}")
    logger.info(f"MODEL: {args.model_type}")
    logger.info(f"DATA: {args.data_folders}")
    logger.info(f"START: {args.train_date}")
    logger.info(f"DAYS: {args.train_n}")
    logger.info(f"num_workers: {num_workers}")
    logger.info(f"pin_memory: {pin_memory}")

    n_folds = 2
    train_days = standardfeeds.get_date_list(args.train_date, int(args.train_n))
    train_days = [x.split(".")[0] for x in train_days]
    for fold_num in range(n_folds):
        logger.info(f"FOLD: {fold_num}")
        model = model_utils.load_model(args.model_folder, args.run_label, args.model_type, fold_num)
        model.model_name = f"{args.model_type}_TUNED-{fold_num}"
        train_dataset = data_loader.NumpyDataset(
            args.data_folders,
            train_days,
            holdout_routes=model.holdout_routes,
            load_in_memory=False,
            include_grid=True if args.model_type.split("_")[-1]=="REALTIME" else False,
            config = model.config
        )
        train_idx = np.random.choice(np.arange(len(train_dataset)), 100)
        train_sampler = SequentialSampler(train_idx)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            sampler=train_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator=accelerator,
            logger=TensorBoardLogger(save_dir=f"{args.model_folder}{args.run_label}", name=model.model_name),
            callbacks=[EarlyStopping(monitor=f"train_loss", min_delta=.0001, patience=5)],
        )
        trainer.fit(model=model, train_dataloaders=train_loader)
    logger.info(f"{model.model_name} TUNING COMPLETE")