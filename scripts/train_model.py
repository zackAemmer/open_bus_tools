import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold

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
    print(f"TRAINING")
    print(f"DATA: {data_folder}")
    print(f"MODEL: {model_type}")
    print(f"NETWORK: {network_name}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    k_fold = KFold(5, shuffle=True, random_state=42)
    train_dataset = data_loader.ContentDataset(data_folder, train_dates, holdout_type='create')

    # print(f"Building grid on fold training data")
    # train_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/train", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    # train_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    # train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    # train_ngrid.build_cell_lookup()
    # train_dataset.grid = train_ngrid

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(train_dataset.__len__()))):
        print("="*30)
        print(f"FOLD: {fold_num}")
        train_dataset.config = data_loader.create_config(train_dataset.data.loc[train_idx])
        model = model_utils.make_model(model_type, fold_num, train_dataset.config, train_dataset.holdout_routes)
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
            persistent_workers=True
        )
        val_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            sampler=val_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            max_epochs=5,
            min_epochs=1,
            accelerator=accelerator,
            logger=TensorBoardLogger(save_dir=f"{model_folder}{network_name}", name=model.model_name),
            callbacks=[EarlyStopping(monitor=f"valid_loss", min_delta=.001, patience=3)],
            # profiler=pl.profilers.AdvancedProfiler(filename='profiler_results'),
            # limit_train_batches=2,
            # limit_val_batches=2,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"TRAINING COMPLETE")