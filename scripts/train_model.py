import argparse
import logging

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
    parser.add_argument('-df', '--data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--train_date', required=True)
    parser.add_argument('-tn', '--train_n', required=True)
    args = parser.parse_args()

    print("="*30)
    print(f"TRAINING")
    print(f"RUN: {args.run_label}")
    print(f"MODEL: {args.model_type}")
    print(f"DATA: {args.data_folders}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    k_fold = KFold(5, shuffle=True, random_state=42)
    train_dates = standardfeeds.get_date_list(args.train_date, int(args.train_n))
    train_data, holdout_routes, train_config = data_loader.load_h5(args.data_folders, train_dates, holdout_routes=data_loader.HOLDOUT_ROUTES)
    train_dataset = data_loader.H5Dataset(train_data)
    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(train_dataset.__len__()))):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = model_utils.make_model(args.model_type, fold_num, train_config, holdout_routes)
        train_dataset.include_grid = model.include_grid
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
            callbacks=[EarlyStopping(monitor=f"valid_loss", min_delta=.0001, patience=5)],
            # profiler=PyTorchProfiler(dirpath="./profiler/", filename=f"{model.model_name}"),
            # limit_train_batches=2,
            # limit_val_batches=2,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"TRAINING COMPLETE")