import datetime
import pickle
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from openbustools import spatial, standardfeeds
from openbustools.traveltime import data_loader


def drop_track_ends(data, id_col, n_drop=1):
    """
    Drops the first and last rows for each unique value in the specified id_col column of the DataFrame.

    Args:
        data (DataFrame): The input DataFrame.
        id_col (str): The column name used to identify unique values.
        n_drop (int): The number of times to drop the first and last rows. Default is 1.

    Returns:
        DataFrame: The modified DataFrame with dropped rows.
    """
    data = data.reset_index(drop=True)
    for _ in range(n_drop):
        data = data.drop(data.groupby(id_col, as_index=False).nth(0).index)
        data = data.drop(data.groupby(id_col, as_index=False).nth(-1).index)
    return data


def shingle(data, min_break, max_break, resample_shingles=False, min_len=None, max_len=None):
    """
    Split a dataframe into random even chunks, with random resampled len.

    Args:
        data (DataFrame): The input DataFrame.
        min_break (int): The minimum number of breaks to be applied to each shingle.
        max_break (int): The maximum number of breaks to be applied to each shingle.
        resample_shingles (bool): Whether to resample the shingles to a random length.
        min_len (int): The minimum length of each shingle.
        max_len (int): The maximum length of each shingle.

    Returns:
        DataFrame: Input data split into shingle IDs assigned to each row.
    """
    # Set initial id based on unique file and trip id
    shingle_lens = data.groupby(['realtime_filename','trip_id']).count()['lat'].values
    shingle_ids = [id.repeat(grouplen) for id, grouplen in zip(np.arange(len(shingle_lens)), shingle_lens)]
    # Break each shingle into a random number of smaller shingles
    shingle_n_chunks = np.random.randint(min_break, max_break, len(shingle_ids))
    shingle_ids = [np.array_split(shingle, n_chunks) for shingle, n_chunks in zip(shingle_ids, shingle_n_chunks)]
    # Start shingle indexing from 0
    shingle_id_counter = 0
    all_shingles = []
    for i in range(len(shingle_ids)):
        for j in range(len(shingle_ids[i])):
            shingle_ids[i][j][:] = shingle_id_counter
            all_shingles.append(shingle_ids[i][j])
            shingle_id_counter += 1
    shingle_ids = np.concatenate(all_shingles)
    shingled_data = data.copy()
    shingled_data['shingle_id'] = shingle_ids
    if resample_shingles:
        # Now resample each shingle to a random length between min and max
        sids, sidxs = np.unique(shingled_data['shingle_id'], return_index=True)
        shingles = np.split(shingled_data[['locationtime','lon','lat']].to_numpy(), sidxs[1:], axis=0)
        resample_lens = np.random.randint(min_len, max_len, len(shingles))
        resampled= []
        for shingle_data, resample_len in zip(shingles, resample_lens):
            resampled.append(spatial.resample_to_len(shingle_data, resample_len))
        resampled_sids = np.repeat(sids, resample_lens).astype(int)
        # Can't interpolate categoricals, so rejoin them after resampling
        cat_lookup = shingled_data[['realtime_filename','trip_id','shingle_id']].drop_duplicates()
        shingled_data = pd.DataFrame(np.concatenate(resampled), columns=['locationtime','lon','lat'])
        shingled_data['shingle_id'] = resampled_sids
        shingled_data = pd.merge(shingled_data, cat_lookup, on='shingle_id').sort_values(['shingle_id','locationtime'])
    return shingled_data


def filter_on_points(data, min_filter_dict):
    """
    Filter the data based on specified minimum and maximum values for each column in min_filter_dict.

    Args:
        data (DataFrame): The input DataFrame.
        min_filter_dict (dict): A dictionary containing column names as keys and tuples of minimum and maximum values.

    Returns:
        DataFrame: The filtered data.
    """
    # If time between points is too long, or distance is too short, or not found in DEM, drop
    for col, (min_val, max_val) in min_filter_dict.items():
        data = data[data[col]>min_val].copy()
        data = data[data[col]<max_val].copy()
    # Re-calculate geometry features w/o missing points
    data['calc_dist_m'], data['calc_bear_d'], data['calc_time_s'] = spatial.calculate_gps_metrics(data, 'lon', 'lat', time_col='locationtime')
    # First pt is dependent on prev trip metrics
    data = data.drop(data.groupby('shingle_id', as_index=False, sort=False).nth(0).index)
    data['calc_speed_m_s'] = data['calc_dist_m'] / data['calc_time_s']
    data['calc_dist_km'] = data['calc_dist_m'] / 1000.0
    return data


def filter_on_tracks(data, filter_dict):
    """
    Filters out invalid tracks guaranteeing clean samples.

    Args:
        data (DataFrame): The input DataFrame.
        filter_dict (dict): A dictionary containing column names as keys and tuples of minimum and maximum values. Tracks with values outside these ranges will be filtered.

    Returns:
        DataFrame: The filtered data.
    """
    toss_ids = []
    # Get tracks with less than 2 points
    pt_counts = data.groupby('shingle_id')[['calc_dist_m']].count()
    toss_ids.extend(list(pt_counts[pt_counts['calc_dist_m']<2].index))
    # Get tracks with invalid points
    for col, (min_val, max_val) in filter_dict.items():
        toss_ids.extend(list(data[data[col]<min_val]['shingle_id']))
        toss_ids.extend(list(data[data[col]>max_val]['shingle_id']))
    # Filter the list of full shingles w/invalid points
    toss_ids = np.unique(toss_ids)
    data = data[~data['shingle_id'].isin(toss_ids)].copy()
    return data


def add_time_features(data, timezone):
    """
    Add time-related features to the given data.

    Args:
        data (DataFrame): The input DataFrame.
        timezone (str): The timezone to convert the timestamps to.

    Returns:
        DataFrame: Input data with added time features.
    """
    data['t'] = pd.to_datetime(data['locationtime'], unit='s', utc=True).dt.tz_convert(timezone)
    data['t_year'] = data['t'].dt.year
    data['t_month'] = data['t'].dt.month
    data['t_day'] = data['t'].dt.day
    data['t_hour'] = data['t'].dt.hour
    data['t_min'] = data['t'].dt.minute
    data['t_sec'] = data['t'].dt.second
    # For embeddings
    data['t_day_of_week'] = data['t'].dt.dayofweek
    data['t_min_of_day'] = (data['t_hour']*60) + data['t_min']
    # For calculating absolute time differences in trips (w/midnight crossover in schedules)
    data['t_sec_of_day'] = data['t'] - datetime.datetime(min(data['t_year']), min(data['t_month']), min(data['t_day']), 0, tzinfo=ZoneInfo(timezone))
    data['t_sec_of_day'] = data['t_sec_of_day'].dt.total_seconds()
    data['t_sec_of_day_start'] = data.groupby('shingle_id')[['t_sec_of_day']].transform('min')
    return data


def add_static_features(data, static_foldername, epsg):
    """
    Add features from a static bus feed to the given data.

    Args:
        data (DataFrame): The input DataFrame.
        static_foldername (Path): Path to the folder containing the static feed info.
        epsg (int): EPSG code for the coordinate system.

    Returns:
        DataFrame: Input data with added static features.
    """
    data['static_foldername'] = str(static_foldername)
    stop_times, stops, trips = standardfeeds.load_standard_static(static_foldername)
    static_data = standardfeeds.combine_static_stops(stop_times, stops, epsg)
    # Filter any realtime trips that are not in the schedule
    data_filtered_static = data.drop(data[~data['trip_id'].isin(static_data.trip_id)].index)
    # Handle case where none of the realtime ids match the schedule
    if len(data_filtered_static) > 0:
        data = data_filtered_static
        data['stop_id'], data['calc_stop_dist_m'], data['stop_sequence'] = standardfeeds.get_scheduled_arrival(data, static_data)
        data = data.merge(static_data[['trip_id','stop_id','stop_sequence','arrival_time','t_sch_sec_of_day','stop_lon','stop_lat']], on=['trip_id','stop_id','stop_sequence'], how='left')
        data = data.merge(trips, on='trip_id', how='left')
        data['calc_stop_dist_km'] = data['calc_stop_dist_m'] / 1000.0
        # Passed stops
        data['passed_stops_n'] = data.groupby('shingle_id')['stop_sequence'].diff().fillna(0)
        # Scheduled time
        # Handle case where trip in data started the day before collection, giving it scheduled times after midnight but sensed times of same day
        # If the trip started same day as collection and crossed over midnight, the times will line up
        # Can't seem to distinguish between these two cases, so assume anything w/scheduled time over 12hrs started day before and subtract 24hrs from schedule
        data['sch_time_s'] = data['t_sch_sec_of_day'] - data['t_sec_of_day_start']
        data.loc[data['sch_time_s'] > 60*60*12, 't_sch_sec_of_day'] = data.loc[data['sch_time_s'] > 60*60*12, 't_sch_sec_of_day'] - 60*60*24
        data['sch_time_s'] = data['t_sch_sec_of_day'] - data['t_sec_of_day_start']
        data['sch_time_s'] = data['sch_time_s'].fillna(0)
    else:
        data['stop_sequence'] = 0
        data['t_sch_sec_of_day'] = 0
        data['calc_stop_dist_m'] = 0.0
        data['calc_stop_dist_km'] = 0.0
        data['route_id'] = 'TripNotFound'
        data['passed_stops_n'] = 0
        data['sch_time_s'] = 0
    return data


def add_cumulative_features(data):
    """
    Add cumulative features to the given data.

    Args:
        data (DataFrame): The input DataFrame.

    Returns:
        DataFrame: Input data with added cumulative features.
    """
    unique_traj = data.groupby('shingle_id')
    data['cumul_time_s'] = unique_traj['calc_time_s'].cumsum()
    data['cumul_dist_km'] = unique_traj['calc_dist_km'].cumsum()
    data['cumul_dist_m'] = data['cumul_dist_km'] * 1000
    data['cumul_passed_stops_n'] = unique_traj['passed_stops_n'].cumsum()
    data['cumul_time_s'] = data.cumul_time_s - unique_traj.cumul_time_s.transform('min')
    data['cumul_dist_km'] = data.cumul_dist_km - unique_traj.cumul_dist_km.transform('min')
    data['cumul_passed_stops_n'] = data.cumul_passed_stops_n - unique_traj.cumul_passed_stops_n.transform('min')
    return data


def extract_training_features(data):
    """
    Extracts training features from the given data.

    Args:
        data (DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the extracted features in savable numpy format:
            data_id (numpy.ndarray): The 'id' column of the data as an array of integers.
            data_n (numpy.ndarray): The numerical feature columns of the data as an array of integers.
            data_c (numpy.ndarray): The categorical feature columns of the data as an array of strings.
    """
    data_id = data[data_loader.SAMPLE_ID].to_numpy().astype('int32')
    data_n = data[data_loader.NUM_FEAT_COLS].to_numpy().astype('float32')
    data_c = data[data_loader.MISC_CAT_FEATS].to_numpy().astype('S30')
    return (data_id, data_n, data_c)


def add_hex_regions(data, embeddings_dir, epsg):
    area, regions, neighbourhood = spatial.load_regions(embeddings_dir)
    # Spatial join the data to the regions
    regions = regions.to_crs(f"epsg:{epsg}").reset_index()[['region_id', 'geometry']]
    data = data.sjoin(regions, how="left", predicate='within').drop(columns=['index_right'])
    assert(data['region_id'].isna().sum()==0) # Check that all points have a region
    return data


def add_osm_embeddings(data, embeddings_dir):
    embeddings_osm = pd.read_pickle(embeddings_dir / "embeddings_osm.pkl")
    embeddings_osm.columns = [f"{i}_osm_embed" for i in embeddings_osm.columns]
    data = pd.merge(data, embeddings_osm, on='region_id', how='left')
    assert(data['0_osm_embed'].isna().sum()==0) # Check that all regions have embeddings
    return data


# def add_gtfs_embeddings(data, static_folder):
#     embeddings_gtfs = pd.read_pickle(static_folder / "spatial_embeddings" / "embeddings_gtfs.pkl")
#     data = data.merge(embeddings_gtfs, on='region_id', how='left')
#     assert(data) # Check that all regions have embeddings
#     return data