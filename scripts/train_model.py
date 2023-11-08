import json
import os
import shutil
import sys
import time

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn import metrics
from sklearn.model_selection import KFold
from tabulate import tabulate
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler

from openbustools.traveltime import data_loader, grids, model_utils
from openbustools import data_utils


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    model_type = sys.argv[1]
    data_folder = sys.argv[2]
    train_date = sys.argv[3]
    train_n = sys.argv[4]
    test_date = sys.argv[5]
    test_n = sys.argv[6]
    train_dates = data_utils.get_date_list(train_date, int(train_n))
    test_dates = data_utils.get_date_list(test_date, int(test_n))

    # grid_s_size=500

    # if network_folder=="kcm/":
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # elif network_folder=="atb/":
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    # else:
    #     holdout_routes=None

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
    print(f"DATA: '{data_folder}'")
    print(f"MODEL: {model_type}")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    # Data loading and fold setup
    train_dataset = data_loader.ContentDataset(data_folder, train_dates)
    test_dataset = data_loader.ContentDataset(data_folder, test_dates)

    # with open(f"{base_folder}deeptte_formatted/train_summary_config.json", "r") as f:
    #     config = json.load(f)

    # print(f"Building grid on fold training data")
    # train_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/train", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    # train_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    # train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    # train_ngrid.build_cell_lookup()
    # train_dataset.grid = train_ngrid

    # print(f"Building grid on fold testing data")
    # test_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/test", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    # test_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    # test_ngrid.add_grid_content(test_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    # test_ngrid.build_cell_lookup()
    # test_dataset.grid = test_ngrid

    k_fold = KFold(5, shuffle=True, random_state=42)
    run_results = []

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(np.arange(train_dataset.__len__()))):
        print("="*30)
        print(f"FOLD: {fold_num}")

        # Declare models
        model = model_utils.make_model(model_type, fold_num)
        # base_model_list, nn_model = utils.make_one_model(model_type)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SequentialSampler(val_idx)
        # model_names = [m.model_name for m in base_model_list]
        # model_names.append(nn_model.model_name)
        # print(f"Model name: {model_names}")
        # print(f"NN model total parameters: {sum(p.numel() for p in nn_model.parameters())}")

        # Keep track of all model performances
        # model_fold_results = {}
        # for x in model_names:
        #     model_fold_results[x] = {
        #         "Labels":[],
        #         "Preds":[]
        #     }
        # for x in range(10000):
        #     z = train_dataset.__getitem__(x)
        # # Total run samples
        # print(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")

        train_loader = DataLoader(train_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, sampler=train_sampler, drop_last=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        val_loader = DataLoader(train_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, sampler=val_sampler, drop_last=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        test_loader = DataLoader(test_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            max_epochs=5,
            min_epochs=1,
            accelerator=accelerator,
            logger=TensorBoardLogger(save_dir=f"./logs/", name=model.model_name),
            callbacks=[EarlyStopping(monitor=f"valid_loss", min_delta=.001, patience=3)],
            profiler=pl.profilers.AdvancedProfiler(filename='profiler_results'),
            # limit_train_batches=2,
            # limit_val_batches=2,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)


        # for b_model in base_model_list:
        #     train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=b_model.collate_fn, sampler=train_sampler, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        #     val_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=b_model.collate_fn, sampler=val_sampler, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        #     test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
        #     print(f"Network {network_folder} Fold {fold_num} Model {b_model.model_name}")
        #     # Train base model on all fold data
        #     b_model.train_time = 0.0
        #     b_model.train(train_loader, config)
        #     print(f"Fold final evaluation for: {b_model.model_name}")
        #     labels, preds = b_model.evaluate(test_loader, config)
        #     model_fold_results[b_model.model_name]["Labels"].extend(list(labels))
        #     model_fold_results[b_model.model_name]["Preds"].extend(list(preds))
        #     data_utils.write_pkl(b_model, f"{base_folder}models/{b_model.model_name}_{fold_num}.pkl")

        # # Train/Test nn model
        # train_dataset.add_grid_features = nn_model.requires_grid
        # test_dataset.add_grid_features = nn_model.requires_grid
        # train_loader = DataLoader(train_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, sampler=train_sampler, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        # val_loader = DataLoader(train_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, sampler=val_sampler, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        # test_loader = DataLoader(test_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
        # t0=time.time()
        # trainer = pl.Trainer(
        #     check_val_every_n_epoch=1,
        #     max_epochs=50,
        #     min_epochs=5,
        #     accelerator=accelerator,
        #     logger=TensorBoardLogger(save_dir=f"{model_folder}logs/", name=nn_model.model_name),
        #     callbacks=[EarlyStopping(monitor=f"valid_loss", min_delta=.001, patience=3)],
        # )
        # trainer.fit(model=nn_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # nn_model.train_time = time.time() - t0
        # preds_and_labels = trainer.predict(model=nn_model, dataloaders=test_loader)
        # preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
        # labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
        # model_fold_results[nn_model.model_name]["Labels"].extend(list(labels))
        # model_fold_results[nn_model.model_name]["Preds"].extend(list(preds))

        # # After all models have trained for this fold, calculate various losses
        # train_times = [x.train_time for x in base_model_list]
        # train_times.append(nn_model.train_time)
        # fold_results = {
        #     "Model_Names": model_names,
        #     "Fold": fold_num,
        #     "All_Losses": [],
        #     "Train_Times": train_times
        # }
        # for mname in fold_results["Model_Names"]:
        #     _ = [mname]
        #     _.append(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]))
        #     _.append(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"])))
        #     _.append(metrics.mean_absolute_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]))
        #     fold_results['All_Losses'].append(_)

        # # Print results of this fold
        # print(tabulate(fold_results['All_Losses'], headers=["Model", "MAPE", "RMSE", "MAE"]))
        # run_results.append(fold_results)

    # # Save full run results
    # data_utils.write_pkl(run_results, f"{model_folder}model_results.pkl")
    # print(f"MODEL RUN COMPLETED {model_folder}")