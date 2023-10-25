import json
import os
import shutil

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from joblib import Parallel, delayed

from openbustools import data_utils


def process_data_parallel(date_list, i, n, **kwargs):
    # Load data from dates
    traces, fail_dates = data_utils.combine_pkl_data(kwargs['raw_data_folder'][n], date_list, kwargs['given_names'][n])
    if len(fail_dates) > 0:
        print(f"Failed to load data for dates: {fail_dates}")
    num_raw_points = len(traces)
    print(f"Chunk {date_list} found {num_raw_points} points...")
    # Clean and transform raw bus data records
    traces = data_utils.shingle(traces, 2, 5)
    traces = data_utils.calculate_trace_df(traces, kwargs['timezone'][n], kwargs['epsg'][n], kwargs['grid_bounds'][n], kwargs['coord_ref_center'][n], kwargs['data_dropout'])
    if not kwargs['skip_gtfs']:
        traces = data_utils.clean_trace_df_w_timetables(traces, kwargs['gtfs_folder'][n], kwargs['epsg'][n], kwargs['coord_ref_center'][n])
        if len(traces)==0:
            print(f"Lost points from {date_list} when trying to match GTFS")
    traces = data_utils.calculate_cumulative_values(traces, kwargs['skip_gtfs'])
    if len(traces)==0:
        print(f"No data remaining after cleaning for dates: {date_list}")
        return (None, None)
    # Dict: Shingle_id; data_file; lines; file; trip_id; route_id
    # Reduce to absolute minimum variables
    if not kwargs['skip_gtfs']:
        traces = traces[[
            "shingle_id",
            "file",
            "trip_id",
            "weekID",
            "timeID",
            "timeID_s",
            "locationtime",
            "lon",
            "lat",
            "x",
            "y",
            "x_cent",
            "y_cent",
            "dist_calc_km",
            "time_calc_s",
            "dist_cumulative_km",
            "time_cumulative_s",
            "speed_m_s",
            "bearing",
            "route_id",
            "stop_x_cent",
            "stop_y_cent",
            "scheduled_time_s",
            "stop_dist_km",
            "passed_stops_n"
        ]].convert_dtypes()
    else:
        traces = traces[[
            "shingle_id",
            "file",
            "trip_id",
            "weekID",
            "timeID",
            "timeID_s",
            "locationtime",
            "lon",
            "lat",
            "x",
            "y",
            "x_cent",
            "y_cent",
            "dist_calc_km",
            "time_calc_s",
            "dist_cumulative_km",
            "time_cumulative_s",
            "speed_m_s",
            "bearing",
            # "route_id",
            # "stop_x_cent",
            # "stop_y_cent",
            # "scheduled_time_s",
            # "stop_dist_km",
            # "passed_stops_n"
        ]].convert_dtypes()
    # Create a lookup that points to lines in the tabular file corresponding to each shingle
    # Save the shingles to the designated network/save file, and return the lookup of shingle information
    # The tabular format is float, so remove string identifier values and keep tham in the lookup
    # First get summary config for normalization
    summary_config = data_utils.get_summary_config(traces, **kwargs)
    # Then get shingle config for saved lines
    if not kwargs['skip_gtfs']:
        shingle_list = traces.groupby("shingle_id")[["trip_id","file","route_id"]].agg([lambda x: x.iloc[0], len]).reset_index().values
        traces = traces.drop(columns=["trip_id","file","route_id"]).values.astype('float32')
    else:
        shingle_list = traces.groupby("shingle_id")[["trip_id","file"]].agg([lambda x: x.iloc[0], len]).reset_index().values
        traces = traces.drop(columns=["trip_id","file"]).values.astype('float32')
    end_idxs = np.cumsum(shingle_list[:,2])
    start_idxs = np.insert(end_idxs, 0, 0)[:-1]
    shingle_config = {}
    for sidx in range(len(shingle_list)):
        vals = shingle_list[sidx,:]
        shingle_config[sidx] = {}
        shingle_config[sidx]['shingle_id'] = vals[0]
        shingle_config[sidx]['trip_id'] = vals[1]
        shingle_config[sidx]['len'] = vals[2]
        shingle_config[sidx]['file'] = vals[3]
        if not kwargs['skip_gtfs']:
            shingle_config[sidx]['route_id'] = vals[5]
        shingle_config[sidx]['start_idx'] = start_idxs[sidx]
        shingle_config[sidx]['end_idx'] = end_idxs[sidx]
        shingle_config[sidx]['network'] = n
        shingle_config[sidx]['file_num'] = i
    print(f"Retained {np.round(len(traces)/num_raw_points, 2)*100}% of original data points; saving and creating configs...")
    with h5py.File(f"{kwargs['base_folder']}deeptte_formatted/{kwargs['train_or_test']}_data_{n}_{i}.h5", 'w') as f:
        f.create_dataset('tabular_data', data=traces)
    return (shingle_config, summary_config)


def clean_data(dates, **kwargs):
    # Clean a set of dates (allocated to training or testing)
    n_j = kwargs['n_jobs']
    if len(dates) < n_j:
        n_j = 2
    print(f"Processing {kwargs['train_or_test']} data from {len(dates)} dates across {n_j} jobs...")
    date_splits = np.array_split(dates, n_j)
    date_splits = [list(x) for x in date_splits]
    # Handle mixed network datasets
    combined_shingle_configs = {}
    combined_summary_configs = {}
    index = 0
    for n in range(len(kwargs['raw_data_folder'])):
        # n indexes the network (for mixed runs), i indexes the date chunk/file to save, and x is the date chunk
        n_w = kwargs['n_workers']
        if len(date_splits) < kwargs['n_workers']:
            n_w = len(date_splits)
        configs = Parallel(n_jobs=n_w)(delayed(process_data_parallel)(x, i, n, **kwargs) for i, x in enumerate(date_splits))
        shingle_configs = [sh for (sh, su) in configs if sh!=None]
        summary_configs = [su for (sh, su) in configs if su!=None]
        for d in shingle_configs:
            combined_shingle_configs.update({i: v for i, v in enumerate(list(d.values()), start=index)})
            index += len(d)
        combined_summary_configs = data_utils.combine_config_list(summary_configs, avoid_dup=True)
    # Save configs
    print(f"Saving {kwargs['train_or_test']} config and shingle lookups...")
    with open(f"{kwargs['base_folder']}deeptte_formatted/{kwargs['train_or_test']}_summary_config.json", mode="a") as out_file:
            json.dump(combined_summary_configs, out_file)
    with open(f"{kwargs['base_folder']}deeptte_formatted/{kwargs['train_or_test']}_shingle_config.json", mode="a") as out_file:
            json.dump(combined_shingle_configs, out_file)


def prepare_run(overwrite, run_name, network_name, train_dates, test_dates, **kwargs):
    """
    Set up the folder and data structure for a set of k-fold validated model runs.
    All run data is copied from the original download directory to the run folder.
    Separate folders are made for the ATB and KCM networks.
    The formatted data is saved in "deeptte_formatted". Since we are benchmarking
    with that model, it is convenient to use the same data format for all models.
    """
    print("="*30)
    print(f"PREPARE RUN: '{run_name}'")
    print(f"NETWORK: '{network_name}'")
    # Create folder structure
    if len(network_name) > 1:
        kwargs['is_mixed'] = True
        network_name = "_".join(network_name)
    else:
        kwargs['is_mixed'] = False
        network_name = network_name[0]
    base_folder = f"./results/{run_name}/{network_name}/"
    if run_name not in os.listdir("./results/"):
        os.mkdir(f"./results/{run_name}")
    if network_name in os.listdir(f"results/{run_name}/") and overwrite:
        shutil.rmtree(base_folder)
    if network_name not in os.listdir(f"results/{run_name}/"):
        os.mkdir(base_folder)
        os.mkdir(f"{base_folder}deeptte_formatted/")
        os.mkdir(f"{base_folder}models/")
        print(f"Created new results folder for '{run_name}'")
    else:
        print(f"Run '{run_name}/{network_name}' folder already exists in 'results/', delete the folder if new run desired.")
        return None
    kwargs['base_folder'] = base_folder
    print(f"Processing training dates...")
    kwargs['train_or_test'] = "train"
    clean_data(train_dates, **kwargs)
    print(f"Processing testing dates...")
    kwargs['train_or_test'] = "test"
    clean_data(test_dates, **kwargs)
    print(f"RUN PREPARATION COMPLETED '{run_name}/{network_name}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    # # DEBUG
    # prepare_run(
    #     overwrite=True,
    #     run_name="debug",
    #     network_name=["kcm"],
    #     train_dates=data_utils.get_date_list("2023_03_15", 3),
    #     test_dates=data_utils.get_date_list("2023_03_21", 3),
    #     n_workers=2,
    #     n_jobs=2,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/kcm_gtfs/"],
    #     raw_data_folder=["./data/kcm_all_new/"],
    #     timezone=["America/Los_Angeles"],
    #     epsg=["32148"],
    #     grid_bounds=[[369903,37911,409618,87758]],
    #     coord_ref_center=[[386910,69022]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=False
    # )
    # prepare_run(
    #     overwrite=True,
    #     run_name="debug",
    #     network_name=["atb"],
    #     train_dates=data_utils.get_date_list("2023_03_15", 3),
    #     test_dates=data_utils.get_date_list("2023_03_21", 3),
    #     n_workers=2,
    #     n_jobs=2,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/atb_gtfs/"],
    #     raw_data_folder=["./data/atb_all_new/"],
    #     timezone=["Europe/Oslo"],
    #     epsg=["32632"],
    #     grid_bounds=[[550869,7012847,579944,7039521]],
    #     coord_ref_center=[[569472,7034350]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=False
    # )
    # # DEBUG MIXED
    # prepare_run(
    #     overwrite=True,
    #     run_name="debug_nosch",
    #     network_name=["kcm","atb"],
    #     train_dates=data_utils.get_date_list("2023_03_15", 3),
    #     test_dates=data_utils.get_date_list("2023_03_21", 3),
    #     n_workers=2,
    #     n_jobs=2,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/kcm_gtfs/","./data/atb_gtfs/"],
    #     raw_data_folder=["./data/kcm_all_new/","./data/atb_all_new/"],
    #     timezone=["America/Los_Angeles","Europe/Oslo"],
    #     epsg=["32148","32632"],
    #     grid_bounds=[[369903,37911,409618,87758],[550869,7012847,579944,7039521]],
    #     coord_ref_center=[[386910,69022],[569472,7034350]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id'],['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=True
    # )
    # prepare_run(
    #     overwrite=True,
    #     run_name="debug_nosch",
    #     network_name=["rut"],
    #     train_dates=data_utils.get_date_list("2023_03_15", 3),
    #     test_dates=data_utils.get_date_list("2023_03_21", 3),
    #     n_workers=2,
    #     n_jobs=2,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/rut_gtfs/"],
    #     raw_data_folder=["./data/rut_all_new/"],
    #     timezone=["Europe/Oslo"],
    #     epsg=["32632"],
    #     grid_bounds=[[589080,6631314,604705,6648420]],
    #     coord_ref_center=[[597427,6642805]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=True
    # )

    # # FULL RUN
    # prepare_run(
    #     overwrite=True,
    #     run_name="full_run",
    #     network_name=["kcm"],
    #     train_dates=data_utils.get_date_list("2023_02_15", 30),
    #     test_dates=data_utils.get_date_list("2023_04_01", 7),
    #     n_workers=2,
    #     n_jobs=8,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/kcm_gtfs/"],
    #     raw_data_folder=["./data/kcm_all_new/"],
    #     timezone=["America/Los_Angeles"],
    #     epsg=["32148"],
    #     grid_bounds=[[369903,37911,409618,87758]],
    #     coord_ref_center=[[386910,69022]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=False
    # )
    # prepare_run(
    #     overwrite=True,
    #     run_name="full_run",
    #     network_name=["atb"],
    #     train_dates=data_utils.get_date_list("2023_02_15", 30),
    #     test_dates=data_utils.get_date_list("2023_04_01", 7),
    #     n_workers=2,
    #     n_jobs=8,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/atb_gtfs/"],
    #     raw_data_folder=["./data/atb_all_new/"],
    #     timezone=["Europe/Oslo"],
    #     epsg=["32632"],
    #     grid_bounds=[[550869,7012847,579944,7039521]],
    #     coord_ref_center=[[569472,7034350]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=False
    # )
    # # FULL RUN MIXED
    # prepare_run(
    #     overwrite=True,
    #     run_name="full_run_nosch",
    #     network_name=["kcm","atb"],
    #     train_dates=data_utils.get_date_list("2023_02_15", 30),
    #     test_dates=data_utils.get_date_list("2023_04_01", 7),
    #     n_workers=2,
    #     n_jobs=8,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/kcm_gtfs/","./data/atb_gtfs/"],
    #     raw_data_folder=["./data/kcm_all_new/","./data/atb_all_new/"],
    #     timezone=["America/Los_Angeles","Europe/Oslo"],
    #     epsg=["32148","32632"],
    #     grid_bounds=[[369903,37911,409618,87758],[550869,7012847,579944,7039521]],
    #     coord_ref_center=[[386910,69022],[569472,7034350]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id'],['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=True
    # )
    # prepare_run(
    #     overwrite=True,
    #     run_name="full_run_nosch",
    #     network_name=["rut"],
    #     train_dates=data_utils.get_date_list("2023_02_15", 30),
    #     test_dates=data_utils.get_date_list("2023_04_01", 7),
    #     n_workers=2,
    #     n_jobs=8,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/rut_gtfs/"],
    #     raw_data_folder=["./data/rut_all_new/"],
    #     timezone=["Europe/Oslo"],
    #     epsg=["32632"],
    #     grid_bounds=[[589080,6631314,604705,6648420]],
    #     coord_ref_center=[[597427,6642805]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=True
    # )

    # # BIG RUN
    # prepare_run(
    #     overwrite=True,
    #     run_name="big_run",
    #     network_name=["kcm"],
    #     train_dates=data_utils.get_date_list("2023_02_15", 120),
    #     test_dates=data_utils.get_date_list("2023_06_15", 7),
    #     n_workers=4,
    #     n_jobs=12,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/kcm_gtfs/"],
    #     raw_data_folder=["./data/kcm_all_new/"],
    #     timezone=["America/Los_Angeles"],
    #     epsg=["32148"],
    #     grid_bounds=[[369903,37911,409618,87758]],
    #     coord_ref_center=[[386910,69022]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=False
    # )
    # prepare_run(
    #     overwrite=True,
    #     run_name="big_run",
    #     network_name=["atb"],
    #     train_dates=data_utils.get_date_list("2023_02_15", 120),
    #     test_dates=data_utils.get_date_list("2023_06_15", 7),
    #     n_workers=4,
    #     n_jobs=12,
    #     data_dropout=0.2,
    #     gtfs_folder=["./data/atb_gtfs/"],
    #     raw_data_folder=["./data/atb_all_new/"],
    #     timezone=["Europe/Oslo"],
    #     epsg=["32632"],
    #     grid_bounds=[[550869,7012847,579944,7039521]],
    #     coord_ref_center=[[569472,7034350]],
    #     given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
    #     skip_gtfs=False
    # )
    # BIG RUN MIXED
    prepare_run(
        overwrite=True,
        run_name="big_run_nosch",
        network_name=["kcm","atb"],
        train_dates=data_utils.get_date_list("2023_02_15", 90),
        test_dates=data_utils.get_date_list("2023_06_15", 7),
        n_workers=4,
        n_jobs=12,
        data_dropout=0.2,
        gtfs_folder=["./data/kcm_gtfs/","./data/atb_gtfs/"],
        raw_data_folder=["./data/kcm_all_new/","./data/atb_all_new/"],
        timezone=["America/Los_Angeles","Europe/Oslo"],
        epsg=["32148","32632"],
        grid_bounds=[[369903,37911,409618,87758],[550869,7012847,579944,7039521]],
        coord_ref_center=[[386910,69022],[569472,7034350]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id'],['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=True
    )
    prepare_run(
        overwrite=True,
        run_name="big_run_nosch",
        network_name=["rut"],
        train_dates=data_utils.get_date_list("2023_02_15", 90),
        test_dates=data_utils.get_date_list("2023_05_15", 7),
        n_workers=4,
        n_jobs=12,
        data_dropout=0.2,
        gtfs_folder=["./data/rut_gtfs/"],
        raw_data_folder=["./data/rut_all_new/"],
        timezone=["Europe/Oslo"],
        epsg=["32632"],
        grid_bounds=[[589080,6631314,604705,6648420]],
        coord_ref_center=[[597427,6642805]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=True
    )