import argparse
from pathlib import Path
import pickle

import lightning.pytorch as pl
import numpy as np
import torch

from openbustools import standardfeeds
from openbustools.traveltime import data_loader
from openbustools.traveltime.models import avg_speed, persistent, schedule


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--model_folder', required=True)
    parser.add_argument('-r', '--run_label', required=True)
    parser.add_argument('-trdf', '--train_data_folders', nargs='+', required=True)
    parser.add_argument('-tedf', '--test_data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--test_date', required=True)
    parser.add_argument('-tn', '--test_n', required=True)
    args = parser.parse_args()

    test_dates = standardfeeds.get_date_list(args.test_date, int(args.test_n))

    print("="*30)
    print(f"EXPERIMENTS")
    print(f"RUN: {args.run_label}")
    print(f"MODEL: HEURISTICS")
    print(f"TRAIN CITY DATA: {args.train_data_folders}")
    print(f"TEST CITY DATA: {args.test_data_folders}")

    res = {'AVGH':{}, 'AVGM':{}, 'PERT':{}, 'SCH':{}}
    n_folds = 5
    for fold_num in range(n_folds):
        print("="*30)
        print(f"FOLD: {fold_num}")
        for mname in res.keys():
            res[mname][fold_num] = {}
        avg_model = pickle.load(open(f"{args.model_folder}{args.network_name}/AVG_{fold_num}.pkl", 'rb'))
        per_tim_model = persistent.PersistentTimeModel('PERT')
        sch_model = schedule.ScheduleModel('SCH')

        print(f"EXPERIMENT: SAME CITY")
        test_dataset = data_loader.DictDataset(args.train_data_folders, test_dates, holdout_type='specify', holdout_routes=avg_model.holdout_routes)
        test_dataset.config = avg_model.config
        preds_and_labels = avg_model.predict(test_dataset, 'h')
        res['AVGH'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = avg_model.predict(test_dataset, 'm')
        res['AVGM'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = per_tim_model.predict(test_dataset)
        res['PERT'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = sch_model.predict(test_dataset)
        res['SCH'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

        print(f"EXPERIMENT: DIFFERENT CITY")
        test_dataset = data_loader.DictDataset(args.test_data_folders, test_dates)
        test_dataset.config = avg_model.config
        preds_and_labels = avg_model.predict(test_dataset, 'h')
        res['AVGH'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = avg_model.predict(test_dataset, 'm')
        res['AVGM'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = per_tim_model.predict(test_dataset)
        res['PERT'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = sch_model.predict(test_dataset)
        res['SCH'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        test_dataset = data_loader.DictDataset(args.train_data_folders, test_dates, holdout_type='specify', only_holdout=True, holdout_routes=avg_model.holdout_routes)
        test_dataset.config = avg_model.config
        preds_and_labels = avg_model.predict(test_dataset, 'h')
        res['AVGH'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = avg_model.predict(test_dataset, 'm')
        res['AVGM'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = per_tim_model.predict(test_dataset)
        res['PERT'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = sch_model.predict(test_dataset)
        res['SCH'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

    p = Path('.') / 'results' / args.network_name
    p.mkdir(exist_ok=True)
    for model_type in res.keys():
        pickle.dump(res, open(f"./results/{args.network_name}/{model_type}.pkl", "wb"))
    print(f"EXPERIMENTS COMPLETE")