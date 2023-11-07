import json
import os
import pickle
import shutil
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from random import sample

import numpy as np
import pandas as pd
import pyproj
import torch
from statsmodels.stats.weightstats import DescrStatsW

from openbustools import spatial
from openbustools.traveltime import data_loader


def get_validation_dates(validation_path):
    """
    Get a list of date strings corresponding to all trace files stored in a folder.
    validation_path: string containing the path to the folder
    Returns: list of date strings
    """
    dates = []
    files = os.listdir(validation_path)
    for file in files:
        labels = file.split("-")
        dates.append(labels[2] + "-" + labels[3] + "-" + labels[4].split("_")[0])
    return dates


def get_date_list(start, n_days):
    """
    Get a list of date strings starting at a given day and continuing for n days.
    start: date string formatted as 'yyyy_mm_dd'
    n_days: int number of days forward to include from start day
    Returns: list of date strings
    """
    year, month, day = start.split("_")
    base = date(int(year), int(month), int(day))
    date_list = [base + timedelta(days=x) for x in range(n_days)]
    return [f"{date.strftime('%Y_%m_%d')}.pkl" for date in date_list]


def combine_config_list(temp, avoid_dup=False):
    summary_config = {}
    for k in temp[0].keys():
        if k[-4:]=="mean":
            values = [x[k] for x in temp]
            weights = [x["n_samples"] for x in temp]
            wtd_mean = float(DescrStatsW(values, weights=weights, ddof=len(weights)).mean)
            summary_config.update({k:wtd_mean})
        elif k[-3:]=="std":
            values = [x[k]**2 for x in temp]
            weights = [x["n_samples"] for x in temp]
            wtd_std = float(np.sqrt(DescrStatsW(values, weights=weights, ddof=len(weights)).mean))
            summary_config.update({k:wtd_std})
        elif k[:1]=="n":
            values = int(np.sum([x[k] for x in temp]))
            summary_config.update({k:values})
        else:
            # Use if key is same values for all configs in list
            if avoid_dup:
                values = temp[0][k]
            else:
                values = [x[k] for x in temp]
            summary_config.update({k:values})
    return summary_config


def load_all_inputs(run_folder, network_folder, n_samples):
    with open(f"{run_folder}{network_folder}/deeptte_formatted/train_summary_config.json") as f:
        summary_config = json.load(f)
    with open(f"{run_folder}{network_folder}/deeptte_formatted/train_shingle_config.json") as f:
        shingle_config = json.load(f)
    train_dataset = data_loader.LoadSliceDataset(f"{run_folder}{network_folder}deeptte_formatted/train", summary_config)
    train_traces = train_dataset.get_all_samples_shingle_accurate(n_samples)
    return {
        "summary_config": summary_config,
        "shingle_config": shingle_config,
        "train_traces": train_traces
    }


# def combine_pkl_data(folder, file_list, given_names):
#     """
#     Load raw feed data stored in a .pkl file to a dataframe. Unify column names and dtypes.
#     This should ALWAYS be used to load the raw bus data from .pkl files, because it unifies the column names and types from different networks.
#     folder: the folder to search in
#     file_list: the file names to read and combine
#     given_names: list of the names of the features in the raw data
#     Returns: a dataframe of all data concatenated together, a column 'file' is added, also a list of all files with no data.
#     """
#     data_list = []
#     no_data_list = []
#     for file in file_list:
#         try:
#             data = load_pkl(folder + "/" + file, is_pandas=True)
#             data['file'] = file
#             # Get unified column names
#             data = data[given_names]
#             data.columns = FEATURE_LOOKUP.keys()
#             # Nwy locationtimes are downloaded as floats and have a decimal point; must go from object through float again to get int
#             data['locationtime'] = data['locationtime'].astype(float)
#             # Get unified data types
#             data = data.astype(FEATURE_LOOKUP)
#             data_list.append(data)
#         except FileNotFoundError:
#             no_data_list.append(file)
#     data = pd.concat(data_list, axis=0)
#     # Critical to ensure data frame is sorted by date, then trip_id, then locationtime
#     data = data.sort_values(['file','trip_id','locationtime'], ascending=True)
#     return data, no_data_list


def shingle(trace_df, min_len, max_len):
    """Split a df into even chunks randomly between min and max length.
    Each split comes from a group representing a trajectory in the dataframe.
    trace_df: dataframe of raw bus data
    min_len: minimum number of chunks to split a trajectory into
    max_lan: maximum number of chunks to split a trajectory into
    Returns: A copy of trace_df with a new index, traces with <=2 points removed.
    """
    shingle_groups = trace_df.groupby(['file','trip_id']).count()['lat'].values
    idx = 0
    new_idx = []
    for num_pts in shingle_groups:
        dummy = np.array([0 for i in range(0,num_pts)])
        dummy = np.array_split(dummy, np.random.randint(min_len, max_len))
        dummy = [len(x) for x in dummy]
        for x in dummy:
            [new_idx.append(idx) for y in range(0,x)]
            idx += 1
    z = trace_df.copy()
    z['shingle_id'] = new_idx
    return z


def calculate_cumulative_values(data, skip_gtfs):
    """
    Calculate values that accumulate across each trajectory.
    """
    unique_traj = data.groupby('shingle_id')
    if not skip_gtfs:
        # Get number of passed stops
        data['passed_stops_n'] = unique_traj['stop_sequence'].diff()
        data['passed_stops_n'] = data['passed_stops_n'].fillna(0)
    # Get cumulative values from trip start
    data['time_cumulative_s'] = unique_traj['time_calc_s'].cumsum()
    data['dist_cumulative_km'] = unique_traj['dist_calc_km'].cumsum()
    data['time_cumulative_s'] = data.time_cumulative_s - unique_traj.time_cumulative_s.transform('min')
    data['dist_cumulative_km'] = data.dist_cumulative_km - unique_traj.dist_cumulative_km.transform('min')
    # Remove shingles that don't traverse more than a kilometer, or have less than n points
    data = data.groupby(['shingle_id']).filter(lambda x: np.max(x.dist_cumulative_km) >= 1.0)
    # Trips that cross over midnight can end up with outlier travel times; there are very few so remove trips over 3hrs
    data = data.groupby(['shingle_id']).filter(lambda x: np.max(x.time_cumulative_s) <= 3000)
    data = data.groupby(['shingle_id']).filter(lambda x: len(x) >= 5)
    return data


def get_summary_config(trace_data, **kwargs):
    """
    Get a dict of means and sds which are used to normalize data by DeepTTE.
    trace_data: pandas dataframe with unified columns and calculated distances
    Returns: dict of mean and std values, as well as train/test filenames.
    """
    grouped = trace_data.groupby('shingle_id')
    if not kwargs['skip_gtfs']:
        summary_dict = {
            # Total trip values
            "time_mean": np.mean(grouped.max()[['time_cumulative_s']].values.flatten()),
            "time_std": np.std(grouped.max()[['time_cumulative_s']].values.flatten()),
            "dist_mean": np.mean(grouped.max()[['dist_cumulative_km']].values.flatten()),
            'dist_std': np.std(grouped.max()[['dist_cumulative_km']].values.flatten()),
            # Individual point values (no cumulative)
            'lon_mean': np.mean(trace_data['lon']),
            'lon_std': np.std(trace_data['lon']),
            'lat_mean': np.mean(trace_data['lat']),
            "lat_std": np.std(trace_data['lat']),
            # Other variables:
            "x_cent_mean": np.mean(trace_data['x_cent']),
            "x_cent_std": np.std(trace_data['x_cent']),
            "y_cent_mean": np.mean(trace_data['y_cent']),
            "y_cent_std": np.std(trace_data['y_cent']),
            "speed_m_s_mean": np.mean(trace_data['speed_m_s']),
            "speed_m_s_std": np.std(trace_data['speed_m_s']),
            "bearing_mean": np.mean(trace_data['bearing']),
            "bearing_std": np.std(trace_data['bearing']),
            # Normalization for both cumulative and individual time/dist values
            "dist_calc_km_mean": np.mean(trace_data['dist_calc_km']),
            "dist_calc_km_std": np.std(trace_data['dist_calc_km']),
            "time_calc_s_mean": np.mean(trace_data['time_calc_s']),
            "time_calc_s_std": np.std(trace_data['time_calc_s']),
            # Nearest stop
            "stop_x_cent_mean": np.mean(trace_data['stop_x_cent']),
            "stop_x_cent_std": np.std(trace_data['stop_x_cent']),
            "stop_y_cent_mean": np.mean(trace_data['stop_y_cent']),
            "stop_y_cent_std": np.std(trace_data['stop_y_cent']),
            "stop_dist_km_mean": np.mean(trace_data['stop_dist_km']),
            "stop_dist_km_std": np.std(trace_data['stop_dist_km']),
            "scheduled_time_s_mean": np.mean(grouped.max()[['scheduled_time_s']].values.flatten()),
            "scheduled_time_s_std": np.std(grouped.max()[['scheduled_time_s']].values.flatten()),
            "passed_stops_n_mean": np.mean(trace_data['passed_stops_n']),
            "passed_stops_n_std": np.std(trace_data['passed_stops_n']),
            # Not variables
            "n_points": len(trace_data),
            "n_samples": len(grouped),
            "gtfs_folder": kwargs['gtfs_folder'],
            "epsg": kwargs['epsg'],
            "grid_bounds": kwargs['grid_bounds'],
            "coord_ref_center": kwargs['coord_ref_center']
        }
    else:
        summary_dict = {
            # Total trip values
            "time_mean": np.mean(grouped.max()[['time_cumulative_s']].values.flatten()),
            "time_std": np.std(grouped.max()[['time_cumulative_s']].values.flatten()),
            "dist_mean": np.mean(grouped.max()[['dist_cumulative_km']].values.flatten()),
            'dist_std': np.std(grouped.max()[['dist_cumulative_km']].values.flatten()),
            # Individual point values (no cumulative)
            'lons_mean': np.mean(trace_data['lon']),
            'lons_std': np.std(trace_data['lon']),
            'lat_mean': np.mean(trace_data['lat']),
            "lat_std": np.std(trace_data['lat']),
            # Other variables:
            "x_cent_mean": np.mean(trace_data['x_cent']),
            "x_cent_std": np.std(trace_data['x_cent']),
            "y_cent_mean": np.mean(trace_data['y_cent']),
            "y_cent_std": np.std(trace_data['y_cent']),
            "speed_m_s_mean": np.mean(trace_data['speed_m_s']),
            "speed_m_s_std": np.std(trace_data['speed_m_s']),
            "bearing_mean": np.mean(trace_data['bearing']),
            "bearing_std": np.std(trace_data['bearing']),
            # Normalization for both cumulative and individual time/dist values
            "dist_calc_km_mean": np.mean(trace_data['dist_calc_km']),
            "dist_calc_km_std": np.std(trace_data['dist_calc_km']),
            "time_calc_s_mean": np.mean(trace_data['time_calc_s']),
            "time_calc_s_std": np.std(trace_data['time_calc_s']),
            # # Nearest stop
            # "stop_x_cent_mean": np.mean(trace_data['stop_x_cent']),
            # "stop_x_cent_std": np.std(trace_data['stop_x_cent']),
            # "stop_y_cent_mean": np.mean(trace_data['stop_y_cent']),
            # "stop_y_cent_std": np.std(trace_data['stop_y_cent']),
            # "stop_dist_km_mean": np.mean(trace_data['stop_dist_km']),
            # "stop_dist_km_std": np.std(trace_data['stop_dist_km']),
            # "scheduled_time_s_mean": np.mean(grouped.max()[['scheduled_time_s']].values.flatten()),
            # "scheduled_time_s_std": np.std(grouped.max()[['scheduled_time_s']].values.flatten()),
            # "passed_stops_n_mean": np.mean(trace_data['passed_stops_n']),
            # "passed_stops_n_std": np.std(trace_data['passed_stops_n']),
            # Not variables
            "n_points": len(trace_data),
            "n_samples": len(grouped),
            "gtfs_folder": kwargs['gtfs_folder'],
            "epsg": kwargs['epsg'],
            "grid_bounds": kwargs['grid_bounds'],
            "coord_ref_center": kwargs['coord_ref_center']
        }
    return summary_dict


def get_dataset_stats(data_folder):
    stats = {}
    file_list = os.listdir(data_folder)
    stats["num_days"] = len(file_list)
    stats["start_day"] = min(file_list)
    stats["end_day"] = max(file_list)
    return stats