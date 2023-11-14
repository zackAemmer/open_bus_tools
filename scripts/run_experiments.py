import argparse
from pathlib import Path
import pickle

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from openbustools.traveltime import data_loader, model_utils
from openbustools import data_utils


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', required=True)
    parser.add_argument('-mf', '--model_folder', required=True)
    parser.add_argument('-n', '--network_name', required=True)
    parser.add_argument('-trdf', '--train_data_folders', nargs='+', required=True)
    parser.add_argument('-tedf', '--test_data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--test_date', required=True)
    parser.add_argument('-tn', '--test_n', required=True)
    args = parser.parse_args()

    test_dates = data_utils.get_date_list(args.test_date, int(args.test_n))

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
    print(f"TRAIN CITY DATA: {args.train_data_folders}")
    print(f"TEST CITY DATA: {args.test_data_folders}")
    print(f"MODEL: {args.model_type}")
    print(f"NETWORK: {args.network_name}")
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
        model = model_utils.load_model(args.model_folder, args.network_name, args.model_type, fold_num)
        res[fold_num] = {}

        print(f"EXPERIMENT: SAME CITY")
        test_dataset = data_loader.ContentDataset(args.train_data_folders, test_dates, holdout_type='specify', holdout_routes=model.holdout_routes)
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
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        preds = data_loader.denormalize(preds, model.config['cumul_time_s'])
        labels = data_loader.denormalize(labels, model.config['cumul_time_s'])
        res[fold_num]['same_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: DIFFERENT CITY")
        test_dataset = data_loader.ContentDataset(args.test_data_folders, test_dates)
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
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        preds = data_loader.denormalize(preds, model.config['cumul_time_s'])
        labels = data_loader.denormalize(labels, model.config['cumul_time_s'])
        res[fold_num]['diff_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        test_dataset = data_loader.ContentDataset(args.train_data_folders, test_dates, holdout_type='specify', only_holdout=True, holdout_routes=model.holdout_routes)
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
        preds = np.concatenate([x['preds'] for x in preds_and_labels])
        labels = np.concatenate([x['labels'] for x in preds_and_labels])
        preds = data_loader.denormalize(preds, model.config['cumul_time_s'])
        labels = data_loader.denormalize(labels, model.config['cumul_time_s'])
        res[fold_num]['holdout'] = {'preds':preds, 'labels':labels}

    p = Path('.') / 'results' / args.network_name
    p.mkdir(exist_ok=True)
    pickle.dump(res, open(f"./results/{args.network_name}/{args.model_type}.pkl", "wb"))
    print(f"EXPERIMENTS COMPLETE")