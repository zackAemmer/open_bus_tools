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
from obt import data_loader
from statsmodels.stats.weightstats import DescrStatsW

from obt import shape_utils


# Set of unified feature names and dtypes for variables in the GTFS-RT data
FEATURE_NAMES = ['trip_id','file','locationtime','lat','lon','vehicle_id']
FEATURE_TYPES = ['object','object','int','float','float','object']
FEATURE_LOOKUP = dict(zip(FEATURE_NAMES, FEATURE_TYPES))

# Set of unified feature names and dtypes for variables in the GTFS data
GTFS_NAMES = ['trip_id','stop_id','stop_lat','stop_lon','arrival_time']
GTFS_TYPES = [str,str,float,float,str]
GTFS_LOOKUP = dict(zip(GTFS_NAMES, GTFS_TYPES))


def load_pkl(path, is_pandas=False):
    with open(path, 'rb') as f:
        if is_pandas:
            data = pd.read_pickle(f)
        else:
            data = pickle.load(f)
    return data

def write_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return None

def calculate_gps_dist(end_x, end_y, start_x, start_y):
    """
    Calculate the euclidean distance between a series of points.
    Returns: array of distances in meters.
    """
    x_diff = (end_x - start_x)
    y_diff = (end_y - start_y)
    dists = np.sqrt(x_diff**2 + y_diff**2)
    # Measured in degrees from the positive x axis
    # E==0, N==90, W==180, S==-90
    bearings = np.arctan2(y_diff, x_diff)*180/np.pi
    return dists, bearings

def calculate_trip_speeds(data):
    """
    Calculate speeds between consecutive trip locations.
    Returns: array of speeds, dist_diff, time_diff between consecutive points.
    Nan for first point of a trip.
    """
    x = data[['shingle_id','x','y','locationtime']]
    y = data[['shingle_id','x','y','locationtime']].shift()
    y.columns = [colname+"_shift" for colname in y.columns]
    z = pd.concat([x,y], axis=1)
    z['dist_diff'], z['bearing'] = calculate_gps_dist(z['x'].values, z['y'].values, z['x_shift'].values, z['y_shift'].values)
    z['time_diff'] = z['locationtime'] - z['locationtime_shift']
    z['speed_m_s'] = z['dist_diff'] / z['time_diff']
    return z['speed_m_s'].values, z['dist_diff'].values, z['time_diff'].values, z['bearing'].values

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

def combine_pkl_data(folder, file_list, given_names):
    """
    Load raw feed data stored in a .pkl file to a dataframe. Unify column names and dtypes.
    This should ALWAYS be used to load the raw bus data from .pkl files, because it unifies the column names and types from different networks.
    folder: the folder to search in
    file_list: the file names to read and combine
    given_names: list of the names of the features in the raw data
    Returns: a dataframe of all data concatenated together, a column 'file' is added, also a list of all files with no data.
    """
    data_list = []
    no_data_list = []
    for file in file_list:
        try:
            data = load_pkl(folder + "/" + file, is_pandas=True)
            data['file'] = file
            # Get unified column names
            data = data[given_names]
            data.columns = FEATURE_LOOKUP.keys()
            # Nwy locationtimes are downloaded as floats and have a decimal point; must go from object through float again to get int
            data['locationtime'] = data['locationtime'].astype(float)
            # Get unified data types
            data = data.astype(FEATURE_LOOKUP)
            data_list.append(data)
        except FileNotFoundError:
            no_data_list.append(file)
    data = pd.concat(data_list, axis=0)
    # Critical to ensure data frame is sorted by date, then trip_id, then locationtime
    data = data.sort_values(['file','trip_id','locationtime'], ascending=True)
    return data, no_data_list

def shingle(trace_df, min_len, max_len):
    """
    Split a df into even chunks randomly between min and max length.
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

def calculate_trace_df(data, timezone, epsg, grid_bounds, coord_ref_center, data_dropout=.10):
    """
    Calculate difference in metrics between two consecutive trip points.
    This is the only place where points are filtered rather than entire shingles.
    data: pandas df with all bus trips
    timezone: string for timezone the data were collected in
    remove_stopeed_pts: whether to include consecutive points with no bus movement
    Returns: combination of original point values, and new _diff values
    """
    # Some points with collection issues
    data = data[data['lat']!=0]
    data = data[data['lon']!=0]
    # Drop out random points from all shingles
    data = data.reset_index()
    drop_indices = np.random.choice(data.index, int(data_dropout*len(data)), replace=False)
    data = data[~data.index.isin(drop_indices)].reset_index()
    # Project to local coordinate system
    default_crs = pyproj.CRS.from_epsg(4326)
    proj_crs = pyproj.CRS.from_epsg(epsg)
    transformer = pyproj.Transformer.from_crs(default_crs, proj_crs, always_xy=True)
    data['x'], data['y'] = transformer.transform(data['lon'], data['lat'])
    # Add coordinates that are translated s.t. CBD is 0,0
    data['x_cent'] = data['x'] - coord_ref_center[0]
    data['y_cent'] = data['y'] - coord_ref_center[1]
    # Drop points outside of the network/grid bounding box
    data = data[data['x']>grid_bounds[0]]
    data = data[data['y']>grid_bounds[1]]
    data = data[data['x']<grid_bounds[2]]
    data = data[data['y']<grid_bounds[3]]
    # Calculate feature values between consecutive points, assign to the latter point
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    # Remove first point of every trip (not shingle), since its features are based on a different trip
    data = data.groupby(['file','trip_id'], as_index=False).apply(lambda group: group.iloc[1:,:])
    # Remove any points which seem to be erroneous or repeated
    data = data[data['dist_calc_m']>0]
    data = data[data['dist_calc_m']<20000]
    data = data[data['time_calc_s']>0]
    data = data[data['time_calc_s']<60*60]
    data = data[data['speed_m_s']>0]
    data = data[data['speed_m_s']<35]
    # Now error points are removed, recalculate time and speed features
    # From here out, must filter shingles in order to not change time/dist calcs
    # Note that any point filtering necessitates recalculating travel times for individual points
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    shingles = data.groupby(['shingle_id'], as_index=False)
    data = shingles.apply(lambda group: group.iloc[1:,:])
    shingle_dists = shingles[['dist_calc_m']].sum()
    shingle_times = shingles[['time_calc_s']].sum()
    # Remove (shingles this time) based on final calculation of speeds, distances, times
    invalid_shingles = []
    # Total distance
    invalid_shingles.append(shingle_dists[shingle_dists['dist_calc_m']<=0].shingle_id)
    invalid_shingles.append(shingle_dists[shingle_dists['dist_calc_m']>=20000].shingle_id)
    # Total time
    invalid_shingles.append(shingle_times[shingle_times['time_calc_s']<=0].shingle_id)
    invalid_shingles.append(shingle_times[shingle_times['time_calc_s']>=3*60*60].shingle_id)
    # Invidiual point distance, time, speed
    invalid_shingles.append(data[data['dist_calc_m']<=0].shingle_id)
    invalid_shingles.append(data[data['dist_calc_m']>=20000].shingle_id)
    invalid_shingles.append(data[data['time_calc_s']<=0].shingle_id)
    invalid_shingles.append(data[data['time_calc_s']>=60*60].shingle_id)
    invalid_shingles.append(data[data['speed_m_s']<=0].shingle_id)
    invalid_shingles.append(data[data['speed_m_s']>=35].shingle_id)
    invalid_shingles = pd.concat(invalid_shingles).values
    data = data[~data['shingle_id'].isin(invalid_shingles)]
    data['dist_calc_km'] = data['dist_calc_m'] / 1000.0
    data = data.dropna()
    # Time values for deeptte
    data['datetime'] = pd.to_datetime(data['locationtime'], unit='s', utc=True).map(lambda x: x.tz_convert(timezone))
    data['dateID'] = (data['datetime'].dt.day)
    data['weekID'] = (data['datetime'].dt.dayofweek)
    # (be careful with these last two as they change across the trajectory)
    data['timeID'] = (data['datetime'].dt.hour * 60) + (data['datetime'].dt.minute)
    data['timeID_s'] = (data['datetime'].dt.hour * 60 * 60) + (data['datetime'].dt.minute * 60) + (data['datetime'].dt.second)
    return data

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

def apply_gtfs_timetables(data, gtfs_data, gtfs_folder_date):
    data['gtfs_folder_date'] = gtfs_folder_date
    # Remove any trips that are not in the GTFS
    data.drop(data[~data['trip_id'].isin(gtfs_data.trip_id)].index, inplace=True)
    # Filter trips with less than n observations
    shingle_counts = data['shingle_id'].value_counts()
    valid_trips = shingle_counts.index[shingle_counts >= 5]
    data = data[data['shingle_id'].isin(valid_trips)]
    # Save start time of first points in trajectories
    first_points = data[['shingle_id','timeID_s']].drop_duplicates('shingle_id')
    first_points.columns = ['shingle_id','trip_start_timeID_s']
    closest_stops = get_scheduled_arrival(
        data['trip_id'].values,
        data['x'].values,
        data['y'].values,
        gtfs_data
    )
    data = data.assign(stop_x=closest_stops[:,0])
    data = data.assign(stop_y=closest_stops[:,1]) # Skip trip_id
    data = data.assign(stop_arrival_s=closest_stops[:,3])
    data = data.assign(stop_sequence=closest_stops[:,4])
    data = data.assign(stop_x_cent=closest_stops[:,5])
    data = data.assign(stop_y_cent=closest_stops[:,6])
    data = data.assign(route_id=closest_stops[:,7])
    data = data.assign(route_id=closest_stops[:,7])
    data = data.assign(service_id=closest_stops[:,8])
    data = data.assign(direction_id=closest_stops[:,9])
    data = data.assign(stop_dist_km=closest_stops[:,10]/1000)
    # Get the timeID_s (for the first point of each trajectory)
    data = pd.merge(data, first_points, on='shingle_id')
    # Calculate the scheduled travel time from the first to each point in the shingle
    data = data.assign(scheduled_time_s=data['stop_arrival_s'] - data['trip_start_timeID_s'])
    # Filter out shingles where the data started after midnight, but the trip started before
    # If the data started before the scheduled time difference is still accurate
    valid_trips = data.groupby('shingle_id').filter(lambda x: x['scheduled_time_s'].max() <= 10000)['shingle_id'].unique()
    data = data[data['shingle_id'].isin(valid_trips)]
    return data

def get_scheduled_arrival(trip_ids, x, y, gtfs_data):
    """
    Find the nearest stop to a set of trip-coordinates, and return the scheduled arrival time.
    trip_ids: list of trip_ids
    lons/lat: lists of places where the bus will be arriving (end point of traj)
    gtfs_data: merged GTFS files
    Returns: (distance to closest stop in km, scheduled arrival time at that stop).
    """
    gtfs_data_ary = gtfs_data[['stop_x','stop_y','trip_id','arrival_s','stop_sequence','stop_x_cent','stop_y_cent','route_id','service_id','direction_id']].values
    gtfs_data_coords = gtfs_data[['stop_x','stop_y']].values.astype(float)
    gtfs_data_trips = gtfs_data[['trip_id']].values.flatten().tolist()
    data_coords = np.column_stack([x, y, np.arange(len(x))]).astype(float)
    # Create dictionary mapping trip_ids to lists of points in gtfs
    id_to_stops = {}
    for i, tripid in enumerate(gtfs_data_trips):
        # If the key does not exist, insert the second argument. Otherwise return the value. Append afterward regardless.
        id_to_stops.setdefault(tripid,[]).append((gtfs_data_coords[i], gtfs_data_ary[i]))
    # Repeat for trips in the data
    id_to_data = {}
    for i, tripid in enumerate(trip_ids):
        id_to_data.setdefault(tripid,[]).append(data_coords[i])
    # Iterate over each unique trip, getting closest stops for all points from that trip, and aggregating
    # Adding closest stop distance, and sequence number to the end of gtfs_data features
    result_counter = 0
    result = np.zeros((len(data_coords), gtfs_data_ary.shape[1]+2), dtype=object)
    for key, value in id_to_data.items():
        trip_data = np.vstack(value)
        stop_data = id_to_stops[key]
        stop_coords = np.vstack([x[0] for x in stop_data])
        stop_feats = np.vstack([x[1] for x in stop_data])
        stop_dists, stop_idxs = shape_utils.get_closest_point(stop_coords[:,:2], trip_data[:,:2])
        result[result_counter:result_counter+len(stop_idxs),:-2] = stop_feats[stop_idxs]
        result[result_counter:result_counter+len(stop_idxs),-2] = stop_dists
        result[result_counter:result_counter+len(stop_idxs),-1] = trip_data[:,-1]
        result_counter += len(stop_idxs)
    # Sort the data points from aggregated trips back into their respective shingles
    original_order = np.argsort(result[:,-1])
    result = result[original_order,:]
    return result

def get_best_gtfs_lookup(traces, gtfs_folder):
    # Get the most recent GTFS files available corresponding to each unique file in the traces
    gtfs_available = [f for f in os.listdir(gtfs_folder) if not f.startswith('.')]
    gtfs_available = [datetime.strptime(x, "%Y_%m_%d") for x in gtfs_available]
    dates_needed_string = list(pd.unique(traces['file']))
    dates_needed = [datetime.strptime(x[:10], "%Y_%m_%d") for x in dates_needed_string]
    best_gtfs_dates = []
    for fdate in dates_needed:
        matching_gtfs = [x for x in gtfs_available if x < fdate]
        best_gtfs = max(matching_gtfs)
        best_gtfs_dates.append(best_gtfs)
    best_gtfs_dates_string = [x.strftime("%Y_%m_%d") for x in best_gtfs_dates]
    file_to_gtfs_map = {k:v for k,v in zip(dates_needed_string, best_gtfs_dates_string)}
    return file_to_gtfs_map

def clean_trace_df_w_timetables(traces, gtfs_folder, epsg, coord_ref_center):
    """
    Validate a set of tracked bus locations against GTFS.
    data: pandas dataframe with unified bus data
    gtfs_data: merged GTFS files
    Returns: dataframe with only trips that are in GTFS, and are reasonably close to scheduled stop ids.
    """
    # Process each chunk of traces using corresponding GTFS files, load 1 set of GTFS at a time
    # The dates should be in order so that each GTFS file set is loaded only once
    # Also seems best to only run the merge once, with as many dates as possible
    file_to_gtfs_map = get_best_gtfs_lookup(traces, gtfs_folder)
    result = []
    unique_gtfs = pd.unique(list(file_to_gtfs_map.values()))
    for current_gtfs_name in unique_gtfs:
        keys = [k for k,v in file_to_gtfs_map.items() if v==current_gtfs_name]
        current_gtfs_data = merge_gtfs_files(f"{gtfs_folder}{current_gtfs_name}/", epsg, coord_ref_center)
        result.append(apply_gtfs_timetables(traces[traces['file'].isin(keys)].copy(), current_gtfs_data, current_gtfs_name))
    result = pd.concat(result)
    result = result.sort_values(["file","shingle_id","locationtime"], ascending=True)
    return result

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

def extract_operator(old_folder, new_folder, source_col, op_name):
    """
    Make a copy of raw bus data with only a single operator.
    old_folder: location of data which contains the desired operator + others
    new_folder: where to save the new filtered files
    source_col: column name to filter on
    op_name: column value to keep in new files
    """
    files = os.listdir(old_folder)
    for file in files:
        print(f"Extracting {op_name} from {old_folder}{file} to {new_folder}{file}...")
        if file != ".DS_Store":
            data = load_pkl(f"{old_folder}{file}", is_pandas=True)
            data = data[data[source_col]==op_name]
            write_pkl(data,f"{new_folder}{file}" )

def extract_operator_gtfs(old_folder, new_folder, source_col_trips, source_col_stop_times, op_name):
    """
    First makes a copy of the GTFS directory, then overwrite the key files.
    Example SCP with ipv6: scp -6 ./2023_04_21.zip zack@\[address incl brackets]:/home/zack/2023_04_21.zip
    """
    gtfs_folders = os.listdir(old_folder)
    for file in gtfs_folders:
        if file != ".DS_Store" and len(file)==10:
            print(f"Extracting {op_name} from {old_folder}{file} to {new_folder}{file}...")
            # Delete and remake folder if already exists
            if file in os.listdir(f"{new_folder}"):
                shutil.rmtree(f"{new_folder}{file}")
            shutil.copytree(f"{old_folder}{file}", f"{new_folder}{file}")
            # Read in and overwrite relevant files in new directory
            z = pd.read_csv(f"{new_folder}{file}/trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
            st = pd.read_csv(f"{new_folder}{file}/stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
            z = z[z[source_col_trips].str[:3]==op_name]
            st = st[st[source_col_stop_times].str[:3]==op_name]
            z.to_csv(f"{new_folder}{file}/trips.txt")
            st.to_csv(f"{new_folder}{file}/stop_times.txt")

def merge_gtfs_files(gtfs_folder, epsg, coord_ref_center):
    """
    Join a set of GTFS files into a single dataframe. Each row is a trip + arrival time.
    Returns: pandas dataframe with merged GTFS data.
    """
    z = pd.read_csv(gtfs_folder+"trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
    st = pd.read_csv(gtfs_folder+"stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
    sl = pd.read_csv(gtfs_folder+"stops.txt", low_memory=False, dtype=GTFS_LOOKUP)
    z = pd.merge(z,st,on="trip_id")
    z = pd.merge(z,sl,on="stop_id")
    # Calculate stop arrival from midnight
    gtfs_data = z.sort_values(['trip_id','stop_sequence'])
    gtfs_data['arrival_s'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in gtfs_data['arrival_time'].str.split(":")]
    # Resequence stops from 0 with increment of 1
    gtfs_data['stop_sequence'] = gtfs_data.groupby('trip_id').cumcount()
    # Project stop locations to local coordinate system
    default_crs = pyproj.CRS.from_epsg(4326)
    proj_crs = pyproj.CRS.from_epsg(epsg)
    transformer = pyproj.Transformer.from_crs(default_crs, proj_crs, always_xy=True)
    gtfs_data['stop_x'], gtfs_data['stop_y'] = transformer.transform(gtfs_data['stop_lon'], gtfs_data['stop_lat'])
    gtfs_data['stop_x_cent'] = gtfs_data['stop_x'] - coord_ref_center[0]
    gtfs_data['stop_y_cent'] = gtfs_data['stop_y'] - coord_ref_center[1]
    return gtfs_data

def extract_results(model_results, city):
    # Extract metric results
    fold_results = [x['All_Losses'] for x in model_results]
    cities = []
    models = []
    mapes = []
    rmses = []
    maes = []
    fold_nums = []
    for fold_num in range(0,len(fold_results)):
        for value in range(0,len(fold_results[0])):
            cities.append(city)
            fold_nums.append(fold_num)
            models.append(fold_results[fold_num][value][0])
            mapes.append(fold_results[fold_num][value][1])
            rmses.append(fold_results[fold_num][value][2])
            maes.append(fold_results[fold_num][value][3])
    result_df = pd.DataFrame({
        "Model": models,
        "City": cities,
        "Fold": fold_nums,
        "MAPE": mapes,
        "RMSE": rmses,
        "MAE": maes
    })
    # # Extract NN loss curves
    # loss_df = []
    # # Iterate folds
    # for fold_results in model_results:
    #     # Iterate models
    #     for model in fold_results['Loss_Curves']:
    #         for mname, loss_curves in model.items():
    #             # Iterate loss curves
    #             for lname, loss in loss_curves.items():
    #                 df = pd.DataFrame({
    #                     "City": city,
    #                     "Fold": fold_results['Fold'],
    #                     "Model": mname,
    #                     "Loss Set": lname,
    #                     "Epoch": np.arange(len(loss)),
    #                     "Loss": loss
    #                 })
    #                 loss_df.append(df)
    # loss_df = pd.concat(loss_df)
    # Extract train times
    names_df = np.array([x['Model_Names'] for x in model_results]).flatten()
    train_time_df = np.array([x['Train_Times'] for x in model_results]).flatten()
    folds_df = np.array([np.repeat(i,len(model_results[i]['Model_Names'])) for i in range(len(model_results))]).flatten()
    city_df = np.array(np.repeat(city,len(folds_df))).flatten()
    train_time_df = pd.DataFrame({
        "City": city_df,
        "Fold": folds_df,
        "Model":  names_df,
        "Time": train_time_df
    })
    return result_df, train_time_df

def extract_gen_results(gen_results, city):
    # Extract generalization results
    res = []
    experiments = ["Train_Losses","Test_Losses","Holdout_Losses","Tune_Train_Losses","Tune_Test_Losses"]
    for ex in experiments:
        fold_results = [x[ex] for x in gen_results]
        cities = []
        models = []
        mapes = []
        rmses = []
        maes = []
        fold_nums = []
        for fold_num in range(0,len(fold_results)):
            for value in range(0,len(fold_results[0])):
                cities.append(city)
                fold_nums.append(fold_num)
                models.append(fold_results[fold_num][value][0])
                mapes.append(fold_results[fold_num][value][1])
                rmses.append(fold_results[fold_num][value][2])
                maes.append(fold_results[fold_num][value][3])
        gen_df = pd.DataFrame({
            "Model": models,
            "City": cities,
            "Loss": ex,
            "Fold": fold_nums,
            "MAPE": mapes,
            "RMSE": rmses,
            "MAE": maes
        })
        res.append(gen_df)
    return pd.concat(res, axis=0)

def extract_lightning_results(model_name, base_folder, city_name):
    all_data = []
    col_names = ["train_loss_epoch","valid_loss","test_loss"]
    # for model_name in os.listdir(base_folder):
    #     model_folder = os.path.join(base_folder, model_name)
    #     if not os.path.isdir(model_folder):
    #         continue
    for fold_folder in os.listdir(base_folder):
        fold_path = os.path.join(base_folder, fold_folder)
        if not os.path.isdir(fold_path):
            continue
        metrics_file = os.path.join(fold_path, "metrics.csv")
        if not os.path.exists(metrics_file):
            continue
        # Read metrics file into a dataframe
        df = pd.read_csv(metrics_file)
        # Rename the columns to include model and fold names
        col_names_mapping = [f"{model_name}_{c}" for c in col_names]
        for i in range(len(col_names)):
            df_sub = df[["epoch", col_names_mapping[i]]].dropna()
            col_remap = {f"{col_names_mapping[i]}": "Loss", "epoch": "Epoch"}
            df_sub.rename(columns=col_remap, inplace=True)
            df_sub["Model"] = model_name
            df_sub["Loss Set"] = col_names[i]
            df_sub["Fold"] = fold_folder.split("_")[1]
            df_sub["City"] = city_name
            df_sub["Loss Set"].replace(to_replace=col_names, value=["Train","Valid","Test"], inplace=True)
            all_data.append(df_sub)
    # Concatenate all dataframes into a single dataframe
    result_df = pd.concat(all_data, axis=0)
    return result_df

def create_tensor_mask(seq_lens, device, drop_first=True):
    """
    Create a mask based on a tensor of sequence lengths.
    """
    max_len = max(seq_lens)
    mask = torch.zeros(len(seq_lens), max_len, dtype=torch.bool, device=device)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1
    if drop_first:
        mask[:,0] = 0
    return mask

def pad_tensors(tensor_list, pad_dim):
    """
    Pad list of tensors with unequal lengths on pad_dim and combine.
    """
    tensor_lens = [tensor.shape[pad_dim] for tensor in tensor_list]
    max_len = max(tensor_lens)
    total_dim = len(tensor_list[0].shape)
    paddings = []
    for tensor in tensor_list:
        padding = list(0 for i in range(total_dim))
        padding[pad_dim] = max_len - tensor.shape[pad_dim]
        paddings.append(tuple(padding))
    padded_tensor_list = [torch.nn.functional.pad(tensor, paddings[i]) for i, tensor in enumerate(tensor_list)]
    padded_tensor_list = torch.cat(padded_tensor_list, dim=0)
    return padded_tensor_list

def aggregate_tts(tts, mask):
    """
    Convert a sequence of predicted travel times to total travel time.
    """
    masked_tts = (tts*mask)
    total_tts = np.sum(masked_tts, axis=1)
    return total_tts

def get_dataset_stats(data_folder):
    stats = {}
    file_list = os.listdir(data_folder)
    stats["num_days"] = len(file_list)
    stats["start_day"] = min(file_list)
    stats["end_day"] = max(file_list)
    return stats

def create_grid_of_shingles(point_resolution, grid_bounds, coord_ref_center):
    # Create grid of coordinates to use as inference inputs
    x_val = np.linspace(grid_bounds[0], grid_bounds[2], point_resolution)
    y_val = np.linspace(grid_bounds[1], grid_bounds[3], point_resolution)
    X,Y = np.meshgrid(x_val,y_val)
    x_spacing = (grid_bounds[2] - grid_bounds[0]) / point_resolution
    y_spacing = (grid_bounds[3] - grid_bounds[1]) / point_resolution
    # Create shingle samples with only geometric variables
    shingles = []
    # All horizontal shingles W-E and E-W
    for i in range(X.shape[0]):
        curr_shingle = {}
        curr_shingle["x"] = list(X[i,:])
        curr_shingle["y"] = list(Y[i,:])
        curr_shingle["dist_calc_km"] = [x_spacing/1000 for x in curr_shingle["x"]]
        curr_shingle["bearing"] = [0 for x in curr_shingle["x"]]
        shingles.append(curr_shingle)
        rev_shingle = {}
        rev_shingle["x"] = list(np.flip(X[i,:]))
        rev_shingle["y"] = list(np.flip(Y[i,:]))
        rev_shingle["dist_calc_km"] = [x_spacing/1000 for x in rev_shingle["x"]]
        rev_shingle["bearing"] = [180 for x in rev_shingle["x"]]
        shingles.append(rev_shingle)
    # All vertical shingles N-S and S-N
    for j in range(X.shape[1]):
        curr_shingle = {}
        curr_shingle["x"] = list(X[:,j])
        curr_shingle["y"] = list(Y[:,j])
        curr_shingle["dist_calc_km"] = [y_spacing/1000 for x in curr_shingle["x"]]
        curr_shingle["bearing"] = [-90 for x in curr_shingle["x"]]
        shingles.append(curr_shingle)
        rev_shingle = {}
        rev_shingle["x"] = list(np.flip(X[:,j]))
        rev_shingle["y"] = list(np.flip(Y[:,j]))
        rev_shingle["dist_calc_km"] = [y_spacing/1000 for x in rev_shingle["x"]]
        rev_shingle["bearing"] = [90 for x in rev_shingle["x"]]
        shingles.append(rev_shingle)
    # Add dummy and calculated variables
    shingle_id = 0
    for curr_shingle in shingles:
        curr_shingle['lat'] = curr_shingle['y']
        curr_shingle['lon'] = curr_shingle['x']
        curr_shingle['shingle_id'] = [shingle_id for x in curr_shingle['lon']]
        curr_shingle['timeID'] = [60*9 for x in curr_shingle['lon']]
        curr_shingle['weekID'] = [3 for x in curr_shingle['lon']]
        curr_shingle['x_cent'] = [x - coord_ref_center[0] for x in curr_shingle['x']]
        curr_shingle['y_cent'] = [y - coord_ref_center[1] for y in curr_shingle['y']]
        curr_shingle['locationtime'] = [1 for x in curr_shingle['lon']]
        curr_shingle['time_calc_s'] = [1 for x in curr_shingle['lon']]
        curr_shingle['dist_cumulative_km'] = [1 for x in curr_shingle['lon']]
        curr_shingle['time_cumulative_s'] = [1 for x in curr_shingle['lon']]
        curr_shingle['speed_m_s'] = [1 for x in curr_shingle['lon']]
        curr_shingle['timeID_s'] = [1 for x in curr_shingle['lon']]
        shingle_id += 1
    # Convert to tabular format, create shingle lookup
    res_dict = {}
    for cname in shingles[0].keys():
        res_dict[cname] = []
        vals = np.array([s[cname] for s in shingles]).flatten()
        res_dict[cname].extend(vals)
    res = pd.DataFrame(res_dict)
    return res