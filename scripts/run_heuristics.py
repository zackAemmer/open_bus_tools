import argparse
import logging
from pathlib import Path
import pickle

import lightning.pytorch as pl
import torch

from openbustools import standardfeeds
from openbustools.traveltime import data_loader
from openbustools.traveltime.models import persistent, schedule


if __name__=="__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('run_experiments')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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
    parser.add_argument('-mf', '--model_folder', required=True)
    parser.add_argument('-r', '--run_label', required=True)
    parser.add_argument('-trdf', '--train_data_folders', nargs='+', required=True)
    parser.add_argument('-tedf', '--test_data_folders', nargs='+', required=True)
    parser.add_argument('-td', '--test_date', required=True)
    parser.add_argument('-tn', '--test_n', required=True)
    args = parser.parse_args()

    logger.info(f"RUN: {args.run_label}")
    logger.info(f"MODEL: HEURISTICS")
    logger.info(f"TRAIN CITY DATA: {args.train_data_folders}")
    logger.info(f"TEST CITY DATA: {args.test_data_folders}")
    logger.info(f"START: {args.test_date}")
    logger.info(f"DAYS: {args.test_n}")
    logger.info(f"num_workers: {num_workers}")
    logger.info(f"pin_memory: {pin_memory}")

    res = {'AVG':{}, 'PERT':{}, 'SCH':{}}
    n_folds = 2
    test_days = standardfeeds.get_date_list(args.test_date, int(args.test_n))
    test_days = [x.split(".")[0] for x in test_days]
    for fold_num in range(n_folds):
        logger.info(f"MODEL: HEURISTICS, FOLD: {fold_num}")
        for mname in res.keys():
            res[mname][fold_num] = {}
        mpath = Path(f"{args.model_folder}{args.run_label}", f"AVG-{fold_num}.pkl")
        avg_model = pickle.load(open(mpath, 'rb'))
        per_tim_model = persistent.PersistentTimeModel('PERT')
        sch_model = schedule.ScheduleModel('SCH')

        logger.info(f"EXPERIMENT: SAME CITY")
        test_dataset = data_loader.NumpyDataset(
            args.train_data_folders,
            test_days,
            holdout_routes=avg_model.holdout_routes,
            load_in_memory=False,
            config=avg_model.config
        )
        preds_and_labels = avg_model.predict(test_dataset)
        res['AVG'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = per_tim_model.predict(test_dataset)
        res['PERT'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = sch_model.predict(test_dataset)
        res['SCH'][fold_num]['same_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

        logger.info(f"EXPERIMENT: DIFFERENT CITY")
        test_dataset = data_loader.NumpyDataset(
            args.test_data_folders,
            test_days,
            holdout_routes=avg_model.holdout_routes,
            load_in_memory=False,
            config=avg_model.config
        )
        preds_and_labels = avg_model.predict(test_dataset)
        res['AVG'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = per_tim_model.predict(test_dataset)
        res['PERT'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = sch_model.predict(test_dataset)
        res['SCH'][fold_num]['diff_city'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

        logger.info(f"EXPERIMENT: HOLDOUT ROUTES")
        test_dataset = data_loader.NumpyDataset(
            args.train_data_folders,
            test_days,
            holdout_routes=avg_model.holdout_routes,
            load_in_memory=False,
            config=avg_model.config,
            only_holdouts=True
        )
        preds_and_labels = avg_model.predict(test_dataset)
        res['AVG'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = per_tim_model.predict(test_dataset)
        res['PERT'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}
        preds_and_labels = sch_model.predict(test_dataset)
        res['SCH'][fold_num]['holdout'] = {'preds':preds_and_labels['preds'], 'labels':preds_and_labels['labels']}

    p = Path("results") / args.run_label
    p.mkdir(parents=True, exist_ok=True)
    for model_name, model_res in res.items():
        pickle.dump(model_res, open(p / f"{model_name}.pkl", 'wb'))
    logger.info(f"HEURISTICS EXPERIMENTS COMPLETE")