import pickle
import logging
from pathlib import Path
import pickle

from dotenv import load_dotenv
load_dotenv()
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import lightning.pytorch as pl
from rasterio.plot import show
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from sklearn.model_selection import KFold

from openbustools import plotting, spatial, standardfeeds
from openbustools.traveltime import data_loader, model_utils
from openbustools.drivecycle import trajectory
from openbustools.drivecycle.physics import conditions, energy, vehicle


def multicity_tuning(**kwargs):
    logger.debug(f"RUNNING MULTICITY TUNING: {kwargs['network_name']}")

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

    save_dir = Path("results","multicity_tuning",kwargs['network_name'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Test base model and heuristic on each city before/after tuning
    res_base = {}
    res_avg = {}
    res_tuned = {}
    res_tuned[row['uuid']] = {}

    model = model_utils.load_model("logs/", kwargs['base_model_network'], kwargs['model_type'], 0)
    model.eval()

    # Test inference for city
    test_dataset = data_loader.NumpyDataset(
        [Path("data","other_feeds",f"{row['uuid']}_realtime","processed")],
        kwargs['test_days'],
        holdout_routes=model.holdout_routes,
        load_in_memory=True,
        config = model.config,
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
        inference_mode=True,
    )
    preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
    preds = np.concatenate([x['preds'] for x in preds_and_labels])
    labels = np.concatenate([x['labels'] for x in preds_and_labels])
    res_base[row['uuid']] = {'preds':preds, 'labels':labels}

    # Load and test heuristic
    model = pickle.load(open(Path("logs", kwargs['base_model_network'], "AVG-0.pkl"), 'rb'))
    preds_and_labels = model.predict(test_dataset)
    res_avg[row['uuid']] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

    # Tune, then re-test the base model on increasing number of data samples
    n_batches = [1, 10, 100, 500, 1000]
    # n_batches = [100]
    batch_size = 10

    for j, batch_limit in enumerate(n_batches):
        # Tune the base model to this city
        model = model_utils.load_model("logs/", kwargs['base_model_network'], kwargs['model_type'], 0)
        model.train()
        train_dataset = data_loader.NumpyDataset(
            [Path("data","other_feeds",f"{row['uuid']}_realtime","processed")],
            kwargs['train_days'],
            load_in_memory=True,
            config=model.config
        )
        k_fold = KFold(5, shuffle=True, random_state=42)
        train_idx, val_idx = list(k_fold.split(np.arange(len(train_dataset))))[0]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SequentialSampler(val_idx)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            train_dataset,
            collate_fn=model.collate_fn,
            batch_size=batch_size,
            sampler=val_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        trainer = pl.Trainer(
            default_root_dir=Path("logs", "other_feeds", f"{row['uuid']}"),
            check_val_every_n_epoch=1,
            max_epochs=100,
            accelerator=accelerator,
            callbacks=[EarlyStopping(monitor=f"valid_loss", min_delta=.0001, patience=3)],
            limit_train_batches=batch_limit,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Test after fine-tuning
        model.eval()
        test_dataset = data_loader.NumpyDataset(
            [Path("data","other_feeds",f"{row['uuid']}_realtime","processed")],
            kwargs['test_days'],
            load_in_memory=True,
            config=model.config
        )
        test_loader = DataLoader(
            test_dataset,
            collate_fn=model.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            logger=False,
            inference_mode=True,
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        res_tuned[row['uuid']][f"{batch_limit}_batches"] = {'preds':preds, 'labels':labels}

    # Save results
    logger.debug(f"SAVING RESULTS: {kwargs['network_name']}")
    filehandler = open(save_dir / "base.pkl", "wb")
    pickle.dump(res_base, filehandler)
    filehandler.close()
    filehandler = open(save_dir / "avg.pkl", "wb")
    pickle.dump(res_avg, filehandler)
    filehandler.close()
    filehandler = open(save_dir / "tuned.pkl", "wb")
    pickle.dump(res_tuned, filehandler)
    filehandler.close()


if __name__=="__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('run_multicity_tuning')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    train_days = standardfeeds.get_date_list('2024_04_03', 4)
    train_days = [x.split(".")[0] for x in train_days]
    test_days = standardfeeds.get_date_list('2024_04_08', 4)
    test_days = [x.split(".")[0] for x in test_days]

    cleaned_sources = pd.read_csv(Path("data", "cleaned_sources.csv"))

    for i, row in cleaned_sources.iloc[:33].iterrows():
        try:
            multicity_tuning(
                base_model_network="kcm",
                model_type="GRU",
                network_name=row['uuid'],
                train_days=train_days,
                test_days=test_days,
            )
        except:
            logger.error(f"ERROR: {row['uuid']}")