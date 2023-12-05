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
        avg_model = pickle.load(open(f"{args.model_folder}{args.run_label}/AVG-{fold_num}.pkl", 'rb'))
        per_tim_model = persistent.PersistentTimeModel('PERT')
        sch_model = schedule.ScheduleModel('SCH')

        print(f"EXPERIMENT: SAME CITY")
        test_data, holdout_routes, test_config = data_loader.load_h5(args.train_data_folders, test_dates, holdout_routes=avg_model.holdout_routes, config=avg_model.config)
        test_dataset = data_loader.H5Dataset(test_data)
        preds_and_labels = avg_model.predict(test_dataset, 'h')
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['AVGH'][fold_num]['same_city'] = {'preds':preds, 'labels':labels}
        preds_and_labels = avg_model.predict(test_dataset, 'm')
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['AVGM'][fold_num]['same_city'] = {'preds':preds, 'labels':labels}
        preds_and_labels = per_tim_model.predict(test_dataset)
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['PERT'][fold_num]['same_city'] = {'preds':preds, 'labels':labels}
        preds_and_labels = sch_model.predict(test_dataset)
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['SCH'][fold_num]['same_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: DIFFERENT CITY")
        test_data, holdout_routes, test_config = data_loader.load_h5(args.test_data_folders, test_dates, holdout_routes=avg_model.holdout_routes, config=avg_model.config)
        test_dataset = data_loader.H5Dataset(test_data)
        preds_and_labels = avg_model.predict(test_dataset, 'h')
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['AVGH'][fold_num]['diff_city'] = {'preds':preds, 'labels':labels}
        preds_and_labels = avg_model.predict(test_dataset, 'm')
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['AVGM'][fold_num]['diff_city'] = {'preds':preds, 'labels':labels}
        preds_and_labels = per_tim_model.predict(test_dataset)
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['PERT'][fold_num]['diff_city'] = {'preds':preds, 'labels':labels}
        preds_and_labels = sch_model.predict(test_dataset)
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['SCH'][fold_num]['diff_city'] = {'preds':preds, 'labels':labels}

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        test_data, holdout_routes, test_config = data_loader.load_h5(args.train_data_folders, test_dates, only_holdout=True, holdout_routes=avg_model.holdout_routes, config=avg_model.config)
        test_dataset = data_loader.H5Dataset(test_data)
        preds_and_labels = avg_model.predict(test_dataset, 'h')
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['AVGH'][fold_num]['holdout'] = {'preds':preds, 'labels':labels}
        preds_and_labels = avg_model.predict(test_dataset, 'm')
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['AVGM'][fold_num]['holdout'] = {'preds':preds, 'labels':labels}
        preds_and_labels = per_tim_model.predict(test_dataset)
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['PERT'][fold_num]['holdout'] = {'preds':preds, 'labels':labels}
        preds_and_labels = sch_model.predict(test_dataset)
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])
        res['SCH'][fold_num]['holdout'] = {'preds':preds, 'labels':labels}
        preds = np.array([x['preds'] for x in preds_and_labels])
        labels = np.array([x['labels'] for x in preds_and_labels])

    p = Path('.') / 'results' / args.run_label
    p.mkdir(exist_ok=True)
    for model_name, model_res in res.items():
        pickle.dump(model_res, open(f"./results/{args.run_label}/{model_name}.pkl", "wb"))
    print(f"EXPERIMENTS COMPLETE")