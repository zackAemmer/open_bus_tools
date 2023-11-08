import json
import shutil
import sys

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn import metrics
from torch.utils.data import DataLoader

from openbustools.traveltime import data_loader, grids, model_utils
from openbustools import data_utils


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    model_type = sys.argv[1]
    run_folder = sys.argv[2]
    train_network_folder = sys.argv[3]
    test_network_folder = sys.argv[4]
    tune_network_folder = sys.argv[5]
    skip_gtfs = sys.argv[6]
    is_param_search = sys.argv[7]

    tune_epochs=5
    grid_s_size=500
    n_tune_samples=100
    n_folds=5

    if skip_gtfs=="True":
        skip_gtfs=True
    else:
        skip_gtfs=False
    if is_param_search=="True":
        is_param_search=True
    else:
        is_param_search=False
    if train_network_folder=="kcm/":
        holdout_routes=[100252,100139,102581,100341,102720]
    elif train_network_folder=="atb/":
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    else:
        holdout_routes=None

    if torch.cuda.is_available():
        num_workers=4
        pin_memory=False
        accelerator="auto"
    else:
        num_workers=0
        pin_memory=False
        accelerator="cpu"

    print("="*30)
    print(f"RUN EXPERIMENTS: '{run_folder}'")
    print(f"MODEL: {model_type}")
    print(f"TRAINED ON NETWORK: '{train_network_folder}'")
    print(f"TUNE ON NETWORK: '{tune_network_folder}'")
    print(f"TEST ON NETWORK: '{test_network_folder}'")
    print(f"num_workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")

    base_folder = f"{run_folder}{train_network_folder}"
    model_folder = f"{run_folder}{train_network_folder}models/{model_type}/"

    try:
        shutil.rmtree(f"{model_folder}gen_logs/")
    except:
        print("Logs folder not found to remove")

    # Define embedded variables for network models
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 8
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 3
        }
    }
    hyperparameter_dict = {
        'FF': {
            'batch_size': 512,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout_rate': .2
        },
        'CONV': {
            'batch_size': 512,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout_rate': .1
        },
        'GRU': {
            'batch_size': 512,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_rate': .05
        },
        'TRSF': {
            'batch_size': 512,
            'hidden_size': 512,
            'num_layers': 6,
            'dropout_rate': .1
        },
        'DEEPTTE': {
            'batch_size': 512
        }
    }

    # Data loading and fold setup
    with open(f"{base_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        train_network_config = json.load(f)
    with open(f"{run_folder}{test_network_folder}deeptte_formatted/test_summary_config.json", "r") as f:
        test_network_config = json.load(f)
    with open(f"{run_folder}{tune_network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        tune_network_config = json.load(f)

    print(f"Building grid on validation data from training network")
    train_network_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/test", train_network_config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    train_network_ngrid = grids.NGridBetter(train_network_config['grid_bounds'][0],grid_s_size)
    train_network_ngrid.add_grid_content(train_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    train_network_ngrid.build_cell_lookup()
    train_network_dataset.grid = train_network_ngrid
    print(f"Building grid on validation data from testing network")
    test_network_dataset = data_loader.LoadSliceDataset(f"{run_folder}{test_network_folder}deeptte_formatted/test", train_network_config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    test_network_ngrid = grids.NGridBetter(test_network_config['grid_bounds'][0],grid_s_size)
    test_network_ngrid.add_grid_content(test_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    test_network_ngrid.build_cell_lookup()
    test_network_dataset.grid = test_network_ngrid
    print(f"Building tune grid on training data from testing network")
    tune_network_dataset = data_loader.LoadSliceDataset(f"{run_folder}{tune_network_folder}deeptte_formatted/train", train_network_config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    tune_network_ngrid = grids.NGridBetter(tune_network_config['grid_bounds'][0],grid_s_size)
    tune_network_ngrid.add_grid_content(tune_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    tune_network_ngrid.build_cell_lookup()
    tune_network_dataset.grid = tune_network_ngrid
    if not skip_gtfs:
        print(f"Building route holdout grid on validation data from training network")
        holdout_network_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/test", train_network_config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs, keep_only_holdout=True)
        holdout_network_ngrid = grids.NGridBetter(train_network_config['grid_bounds'][0],grid_s_size)
        holdout_network_ngrid.add_grid_content(train_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
        holdout_network_ngrid.build_cell_lookup()
        holdout_network_dataset.grid = holdout_network_ngrid

    run_results = []

    for fold_num in range(n_folds):
        # Declare models
        base_model_list, nn_model = model_utils.make_one_model(model_type, hyperparameter_dict=hyperparameter_dict, embed_dict=embed_dict, config=train_network_config, skip_gtfs=skip_gtfs, load_weights=True, weight_folder=f"{model_folder}logs/{model_type}/version_{fold_num}/checkpoints/", fold_num=fold_num)
        model_names = [m.model_name for m in base_model_list]
        model_names.append(nn_model.model_name)
        print(f"Model name: {model_names}")
        print(f"NN model total parameters: {sum(p.numel() for p in nn_model.parameters())}")

        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Train_Labels":[],
                "Train_Preds":[],
                "Test_Labels":[],
                "Test_Preds":[],
                "Holdout_Labels":[],
                "Holdout_Preds":[],
                "Tune_Train_Labels":[],
                "Tune_Train_Preds":[],
                "Tune_Test_Labels":[],
                "Tune_Test_Preds":[]
            }

        print(f"EXPERIMENT: SAME NETWORK")
        for b_model in base_model_list:
            print(f"Fold final evaluation for: {b_model.model_name}")
            loader = DataLoader(train_network_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
            labels, preds = b_model.evaluate(loader, train_network_config)
            model_fold_results[b_model.model_name]["Train_Labels"].extend(list(labels))
            model_fold_results[b_model.model_name]["Train_Preds"].extend(list(preds))
        print(f"Fold final evaluation for: {nn_model.model_name}")
        train_network_dataset.add_grid_features = nn_model.requires_grid
        loader = DataLoader(train_network_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
        trainer = pl.Trainer(accelerator=accelerator, limit_predict_batches=10)
        preds_and_labels = trainer.predict(model=nn_model, dataloaders=loader)
        preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
        labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
        model_fold_results[nn_model.model_name]["Train_Labels"].extend(list(labels))
        model_fold_results[nn_model.model_name]["Train_Preds"].extend(list(preds))

        print(f"EXPERIMENT: DIFFERENT NETWORK")
        for b_model in base_model_list:
            print(f"Fold final evaluation for: {b_model.model_name}")
            loader = DataLoader(test_network_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
            labels, preds = b_model.evaluate(loader, train_network_config)
            model_fold_results[b_model.model_name]["Test_Labels"].extend(list(labels))
            model_fold_results[b_model.model_name]["Test_Preds"].extend(list(preds))
        print(f"Fold final evaluation for: {nn_model.model_name}")
        test_network_dataset.add_grid_features = nn_model.requires_grid
        loader = DataLoader(test_network_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
        trainer = pl.Trainer(accelerator=accelerator, limit_predict_batches=10)
        preds_and_labels = trainer.predict(model=nn_model, dataloaders=loader)
        preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
        labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
        model_fold_results[nn_model.model_name]["Test_Labels"].extend(list(labels))
        model_fold_results[nn_model.model_name]["Test_Preds"].extend(list(preds))

        if not skip_gtfs:
            print(f"EXPERIMENT: HOLDOUT ROUTES")
            for b_model in base_model_list:
                print(f"Fold final evaluation for: {b_model.model_name}")
                loader = DataLoader(holdout_network_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
                labels, preds = b_model.evaluate(loader, train_network_config)
                model_fold_results[b_model.model_name]["Holdout_Labels"].extend(list(labels))
                model_fold_results[b_model.model_name]["Holdout_Preds"].extend(list(preds))
            print(f"Fold final evaluation for: {nn_model.model_name}")
            holdout_network_dataset.add_grid_features = nn_model.requires_grid
            loader = DataLoader(holdout_network_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
            trainer = pl.Trainer(accelerator=accelerator, limit_predict_batches=10)
            preds_and_labels = trainer.predict(model=nn_model, dataloaders=loader)
            preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
            labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
            model_fold_results[nn_model.model_name]["Holdout_Labels"].extend(list(labels))
            model_fold_results[nn_model.model_name]["Holdout_Preds"].extend(list(preds))

            print(f"EXPERIMENT: FINE TUNING")
            # Re-declare models with original weights
            base_model_list, nn_model = model_utils.make_one_model(model_type, hyperparameter_dict=hyperparameter_dict, embed_dict=embed_dict, config=train_network_config, skip_gtfs=skip_gtfs, load_weights=True, weight_folder=f"{model_folder}logs/{model_type}/version_{fold_num}/checkpoints/", fold_num=fold_num)
            for b_model in base_model_list:
                print(f"Fold final evaluation for: {b_model.model_name}")
                loader = DataLoader(train_network_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
                labels, preds = b_model.evaluate(loader, train_network_config)
                model_fold_results[b_model.model_name]["Tune_Train_Labels"].extend(list(labels))
                model_fold_results[b_model.model_name]["Tune_Train_Preds"].extend(list(preds))
                loader = DataLoader(test_network_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
                labels, preds = b_model.evaluate(loader, train_network_config)
                model_fold_results[b_model.model_name]["Tune_Test_Labels"].extend(list(labels))
                model_fold_results[b_model.model_name]["Tune_Test_Preds"].extend(list(preds))
            print(f"Fold training for: {nn_model.model_name}")
            tune_network_dataset.add_grid_features = nn_model.requires_grid
            train_network_dataset.add_grid_features = nn_model.requires_grid
            loader = DataLoader(tune_network_dataset, batch_size=10, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(train_network_dataset, batch_size=10, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
            trainer = pl.Trainer(
                check_val_every_n_epoch=1,
                max_epochs=tune_epochs,
                min_epochs=1,
                limit_train_batches=10,
                limit_val_batches=10,
                logger=TensorBoardLogger(save_dir=f"{model_folder}gen_logs/", name=f"{nn_model.model_name}_TUNE"),
                accelerator=accelerator
            )
            trainer.fit(model=nn_model, train_dataloaders=loader, val_dataloaders=val_loader)
            print(f"Fold final evaluation for: {nn_model.model_name}")
            train_network_dataset.add_grid_features = nn_model.requires_grid
            loader = DataLoader(train_network_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
            trainer = pl.Trainer(accelerator=accelerator, limit_predict_batches=10)
            preds_and_labels = trainer.predict(model=nn_model, dataloaders=loader)
            preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
            labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
            model_fold_results[nn_model.model_name]["Tune_Train_Labels"].extend(list(labels))
            model_fold_results[nn_model.model_name]["Tune_Train_Preds"].extend(list(preds))

            test_network_dataset.add_grid_features = nn_model.requires_grid
            loader = DataLoader(test_network_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
            trainer = pl.Trainer(accelerator=accelerator, limit_predict_batches=10)
            preds_and_labels = trainer.predict(model=nn_model, dataloaders=loader)
            preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
            labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
            model_fold_results[nn_model.model_name]["Tune_Test_Labels"].extend(list(labels))
            model_fold_results[nn_model.model_name]["Tune_Test_Preds"].extend(list(preds))

        # Calculate various losses:
        fold_results = {
            "Model_Names": model_names,
            "Fold": fold_num,
            "Train_Losses": [],
            "Test_Losses": [],
            "Holdout_Losses": [],
            "Tune_Train_Losses": [],
            "Tune_Test_Losses": [],
            "Extract_Train_Losses": [],
            "Extract_Test_Losses": []
        }
        for mname in fold_results["Model_Names"]:
            _ = [mname]
            _.append(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"]))
            _.append(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"])))
            _.append(metrics.mean_absolute_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"]))
            fold_results['Train_Losses'].append(_)
            _ = [mname]
            _.append(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"]))
            _.append(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"])))
            _.append(metrics.mean_absolute_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"]))
            fold_results['Test_Losses'].append(_)
            if not skip_gtfs:
                _ = [mname]
                _.append(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Holdout_Labels"], model_fold_results[mname]["Holdout_Preds"]))
                _.append(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Holdout_Labels"], model_fold_results[mname]["Holdout_Preds"])))
                _.append(metrics.mean_absolute_error(model_fold_results[mname]["Holdout_Labels"], model_fold_results[mname]["Holdout_Preds"]))
                fold_results['Holdout_Losses'].append(_)
                _ = [mname]
                _.append(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Tune_Train_Labels"], model_fold_results[mname]["Tune_Train_Preds"]))
                _.append(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Tune_Train_Labels"], model_fold_results[mname]["Tune_Train_Preds"])))
                _.append(metrics.mean_absolute_error(model_fold_results[mname]["Tune_Train_Labels"], model_fold_results[mname]["Tune_Train_Preds"]))
                fold_results['Tune_Train_Losses'].append(_)
                _ = [mname]
                _.append(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Tune_Test_Labels"], model_fold_results[mname]["Tune_Test_Preds"]))
                _.append(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Tune_Test_Labels"], model_fold_results[mname]["Tune_Test_Preds"])))
                _.append(metrics.mean_absolute_error(model_fold_results[mname]["Tune_Test_Labels"], model_fold_results[mname]["Tune_Test_Preds"]))
                fold_results['Tune_Test_Losses'].append(_)

        # Save fold
        run_results.append(fold_results)

    # Save run results
    data_utils.write_pkl(run_results, f"{model_folder}model_generalization_results.pkl")
    print(f"EXPERIMENTS COMPLETED {model_folder}")