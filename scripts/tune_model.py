import argparse

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

    if torch.cuda.is_available():
        num_workers=4
        pin_memory=True
        accelerator="cuda"
    else:
        num_workers=0
        pin_memory=False
        accelerator="cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', required=True)
    parser.add_argument('-mf', '--model_folder', required=True)
    parser.add_argument('-r', '--run_label', required=True)
    parser.add_argument('-df', '--data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--train_date', required=True)
    parser.add_argument('-tn', '--train_n', required=True)
    args = parser.parse_args()

    train_dates = standardfeeds.get_date_list(args.train_date, int(args.train_n))

    print("="*30)
    print(f"TUNING")
    print(f"RUN: {args.run_label}")
    print(f"MODEL: {args.model_type}")
    print(f"DATA: {args.data_folders}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    n_folds = 5
    for fold_num in range(n_folds):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = model_utils.load_model(args.model_folder, args.run_label, args.model_type, fold_num)
        model.model_name = f"{args.model_type}_TUNED-{fold_num}"
        train_data, holdout_routes, train_config = data_loader.load_h5(args.data_folders, train_dates, config=model.config)
        train_dataset = data_loader.H5Dataset(train_data)
        train_idx = np.random.choice(np.arange(train_dataset.__len__()), 100)
        train_dataset.include_grid = model.include_grid
        train_sampler = SequentialSampler(train_idx)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            sampler=train_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        trainer = pl.Trainer(
            max_epochs=50,
            min_epochs=5,
            accelerator=accelerator,
            logger=TensorBoardLogger(save_dir=f"{args.model_folder}{args.run_label}", name=model.model_name),
            callbacks=[EarlyStopping(monitor=f"train_loss", min_delta=.0001, patience=3)],
        )
        trainer.fit(model=model, train_dataloaders=train_loader)
    print(f"TUNING COMPLETE")