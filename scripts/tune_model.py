import sys

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from tabulate import tabulate
from torch.utils.data import DataLoader, SequentialSampler

from openbustools.traveltime import data_loader, model_utils
from openbustools import data_utils


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    model_type = sys.argv[1]
    model_folder = sys.argv[2]
    network_name = sys.argv[3]
    data_folder = sys.argv[4]
    train_date = sys.argv[5]
    train_n = sys.argv[6]
    train_dates = data_utils.get_date_list(train_date, int(train_n))

    if torch.cuda.is_available():
        num_workers=4
        pin_memory=True
        accelerator="auto"
    else:
        num_workers=4
        pin_memory=False
        accelerator="cpu"

    print("="*30)
    print(f"TUNING")
    print(f"DATA: {data_folder}")
    print(f"MODEL: {model_type}")
    print(f"NETWORK: {network_name}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    # print(f"Building grid on fold training data")
    # train_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/train", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    # train_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    # train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    # train_ngrid.build_cell_lookup()
    # train_dataset.grid = train_ngrid

    n_folds = 5
    for fold_num in range(n_folds):
        print("="*30)
        print(f"FOLD: {fold_num}")
        model = model_utils.load_model(model_folder, network_name, model_type, fold_num)
        model.model_name = f"{model_type}TUNED_{fold_num}"
        train_dataset = data_loader.ContentDataset(data_folder, train_dates)
        train_idx = np.random.choice(np.arange(train_dataset.__len__()), 100)
        train_dataset.config = model.config
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
            max_epochs=10,
            min_epochs=5,
            accelerator=accelerator,
            logger=TensorBoardLogger(save_dir=f"{model_folder}{network_name}", name=model.model_name),
            callbacks=[EarlyStopping(monitor=f"train_loss", min_delta=.001, patience=3)],
        )
        trainer.fit(model=model, train_dataloaders=train_loader)
    print(f"TUNING COMPLETE")