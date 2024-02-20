import argparse
import logging
from pathlib import Path
import shutil

import lightning.pytorch as pl
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from sklearn.model_selection import KFold

from openbustools import standardfeeds
from openbustools.traveltime import data_loader, model_utils


if __name__=="__main__":
    logger = logging.getLogger('train_model')
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
    parser.add_argument('-rs', '--random_seed', required=False)
    parser.add_argument('-hs', '--hyperparameter_search', required=False)
    parser.add_argument('-hsr', '--hyperparameter_search_rmdir', required=False)
    args = parser.parse_args()

    logger.info(f"RUN: {args.run_label}")
    logger.info(f"MODEL: {args.model_type}")
    logger.info(f"DATA: {args.data_folders}")
    logger.info(f"START: {args.train_date}")
    logger.info(f"DAYS: {args.train_n}")
    logger.info(f"num_workers: {num_workers}")
    logger.info(f"pin_memory: {pin_memory}")

    if args.random_seed is not None:
        pl.seed_everything(int(args.random_seed), workers=True)
    else:
        pl.seed_everything(42, workers=True)

    if args.hyperparameter_search_rmdir is not None:
        assert(f"{args.model_folder}{args.run_label}" != "./logs/kcm/")
        assert(f"{args.model_folder}{args.run_label}" != "./logs/atb/")
        save_dir = Path(f"{args.model_folder}{args.run_label}")
        # If exists, delete the results folder
        if save_dir.exists():
            # Wipe the results folder if doing a hyperparameter search
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=False)

    if args.hyperparameter_search is not None:
        hyperparameter_dict = model_utils.sample_hyperparameters()
    else:
        hyperparameter_dict = None

    k_fold = KFold(5, shuffle=True, random_state=42)
    train_days = standardfeeds.get_date_list(args.train_date, int(args.train_n))
    train_days = [x.split(".")[0] for x in train_days]
    train_dataset = data_loader.NumpyDataset(
        args.data_folders,
        train_days,
        holdout_routes=data_loader.HOLDOUT_ROUTES,
        load_in_memory=False,
        include_grid=True if "REALTIME" in args.model_type.split("_") else False
    )
    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(len(train_dataset)))):
        logger.info(f"MODEL: {args.model_type}, FOLD: {fold_num}")
        model = model_utils.make_model(args.model_type, fold_num, train_dataset.config, train_dataset.holdout_routes, hyperparameter_dict)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SequentialSampler(val_idx)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            sampler=val_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            max_epochs=100,
            accelerator=accelerator,
            logger=TensorBoardLogger(save_dir=f"{args.model_folder}{args.run_label}", name=model.model_name),
            callbacks=[EarlyStopping(monitor=f"valid_loss", min_delta=.0001, patience=3)],
            # profiler=PyTorchProfiler(dirpath="./profiler/", filename=f"{model.model_name}"),
            # limit_train_batches=2,
            # limit_val_batches=2,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info(f"{model.model_name} TRAINING COMPLETE")