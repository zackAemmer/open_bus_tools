import datetime
from zoneinfo import ZoneInfo

import geopandas as gpd
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from openbustools import data_utils, spatial, standardfeeds


def prepare_run(**kwargs):
    """Pre-process training data and save to sub-folder."""
    print(f"PROCESSING: {kwargs['network_name']}")
    for day in kwargs['dates']:
        print(day)
        # Loading data and unifying column names/dtypes
        try:
            data = pd.read_pickle(f"{kwargs['realtime_folder']}/{day}")
        except:
            print(f"File failed to load: {day}")
            continue
        num_pts_initial = len(data)
        data['file'] = day[:10]
        data = data[kwargs['given_names']]
        data.columns = standardfeeds.GTFSRT_LOOKUP.keys()
        data['locationtime'] = data['locationtime'].astype(float)
        data = data.astype(standardfeeds.GTFSRT_LOOKUP)
        # Sensors seem to hold old positions right at start/end of trip
        for _ in range(4):
            data = data.drop(data.groupby('trip_id', as_index=False).nth(0).index)
            data = data.drop(data.groupby('trip_id', as_index=False).nth(-1).index)
        # Avoid sensors recording at regular intervals
        drop_indices = np.random.choice(data.index, int(kwargs['data_dropout']*len(data)), replace=False)
        data = data[~data.index.isin(drop_indices)].reset_index().drop(columns='index')
        # Split full trip trajectories into smaller samples
        data = spatial.shingle(data, 2, 5)
        # Project to local coordinate system, apply bounding box, center coords
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs="EPSG:4326").to_crs(f"EPSG:{kwargs['epsg']}")
        data = data.cx[kwargs['grid_bounds'][0]:kwargs['grid_bounds'][2], kwargs['grid_bounds'][1]:kwargs['grid_bounds'][3]].copy()
        data['x_cent'] = data.geometry.x - kwargs['coord_ref_center'][0]
        data['y_cent'] = data.geometry.y - kwargs['coord_ref_center'][1]
        # Calculate geometry features
        data['calc_time_s'], data['calc_dist_m'], data['calc_bear_d'] = spatial.calculate_gps_metrics(data, 'locationtime')
        # Drop consecutive points where bus did not move, re-calculate features
        data = data[data['calc_dist_m']>0].copy()
        data['calc_time_s'], data['calc_dist_m'], data['calc_bear_d'] = spatial.calculate_gps_metrics(data, 'locationtime')
        # First pt of each trip (not shingle) is dependent on prev trip metrics
        data = data.drop(data.groupby('trip_id', as_index=False).nth(0).index)
        data['calc_speed_m_s'] = data['calc_dist_m'] / data['calc_time_s']
        data['calc_dist_km'] = data['calc_dist_m'] / 1000.0
        # Filter trips with less than 2 points
        toss_ids = []
        pt_counts = data.groupby('shingle_id')[['calc_dist_m']].count()
        toss_ids.extend(list(pt_counts[pt_counts['calc_dist_m']<2].index))
        # Filter trips with outliers
        mins_keep = 4
        toss_ids.extend(list(data[data['calc_speed_m_s']>30]['shingle_id']))
        toss_ids.extend(list(data[data['calc_dist_m']>mins_keep*60*30]['shingle_id']))
        toss_ids.extend(list(data[data['calc_time_s']>mins_keep*60]['shingle_id']))
        toss_ids = np.unique(toss_ids)
        data = data[~data['shingle_id'].isin(toss_ids)].copy()
        # Calculate time features
        data['t'] = pd.to_datetime(data['locationtime'], unit='s', utc=True).dt.tz_convert(kwargs['timezone'])
        data['t_year'] = data['t'].dt.year
        data['t_month'] = data['t'].dt.month
        data['t_day'] = data['t'].dt.day
        data['t_hour'] = data['t'].dt.hour
        data['t_min'] = data['t'].dt.minute
        data['t_sec'] = data['t'].dt.second
        # For embeddings
        data['t_day_of_week'] = data['t'].dt.dayofweek
        data['t_min_of_day'] = (data['t_hour']*60) + data['t_min']
        # For calculating absolute time differences in trips (midnight crossover)
        data['t_sec_of_day'] = data['t'] - datetime.datetime(min(data['t_year']), min(data['t_month']), min(data['t_day']), 0, tzinfo=ZoneInfo(kwargs['timezone']))
        data['t_sec_of_day'] = data['t_sec_of_day'].dt.total_seconds()
        # Get GTFS features
        best_static = standardfeeds.latest_available_static(day[:10], kwargs['static_folder'])
        data['best_static'] = best_static
        stops = pd.read_csv(f"{kwargs['static_folder']}{best_static}/stops.txt", low_memory=False, dtype=standardfeeds.GTFS_LOOKUP)[['stop_id','stop_lon','stop_lat']].sort_values('stop_id')
        stop_times = pd.read_csv(f"{kwargs['static_folder']}{best_static}/stop_times.txt", low_memory=False, dtype=standardfeeds.GTFS_LOOKUP)[['trip_id','stop_id','arrival_time','stop_sequence']]
        trips = pd.read_csv(f"{kwargs['static_folder']}{best_static}/trips.txt", low_memory=False, dtype=standardfeeds.GTFS_LOOKUP)[['trip_id','service_id','route_id','direction_id']]
        # Deal with schedule crossing midnight
        stop_times['t_sch_hour'] = stop_times['arrival_time'].str.slice(0,2).astype(int)
        stop_times['t_sch_min'] = stop_times['arrival_time'].str.slice(3,5).astype(int)
        stop_times['t_sch_sec'] = stop_times['arrival_time'].str.slice(7,9).astype(int)
        stop_times['t_sch_min_of_day'] = (stop_times['t_sch_hour']*60) + stop_times['t_sch_min']
        stop_times['t_sch_sec_of_day'] = (stop_times['t_sch_hour']*60*60) + (stop_times['t_sch_min']*60) + stop_times['t_sch_sec']
        stop_times = stop_times.sort_values(['trip_id','t_sch_sec_of_day'])
        stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount()
        static = stop_times.merge(stops, on='stop_id').sort_values(['trip_id','stop_sequence'])
        static = gpd.GeoDataFrame(static, geometry=gpd.points_from_xy(static.stop_lon, static.stop_lat), crs="EPSG:4326").to_crs(f"EPSG:{kwargs['epsg']}")
        # Filter any realtime trips that are not in the schedule
        data.drop(data[~data['trip_id'].isin(static.trip_id)].index, inplace=True)
        data['stop_id'], data['calc_stop_dist_m'], data['stop_sequence'] = standardfeeds.get_scheduled_arrival(data, static)
        # data = data.reset_index().drop(columns='index')
        data = data.merge(stop_times, on=['trip_id','stop_id','stop_sequence'], how='left')
        data = data.merge(trips, on='trip_id', how='left')
        data['calc_stop_dist_km'] = data['calc_stop_dist_m'] / 1000.0
        # Passed stops
        data['pass_stops_n'] = data.groupby('shingle_id')['stop_sequence'].diff()
        data['pass_stops_n'] = data['pass_stops_n'].fillna(0).clip(lower=0)
        # Scheduled time
        data['t_sec_of_day_start'] = data.groupby('shingle_id')[['t_sec_of_day']].transform('min')
        data['sch_time_s'] = data['t_sch_sec_of_day'] - data['t_sec_of_day_start']
        # Cumulative values
        unique_traj = data.groupby('shingle_id')
        data['cumul_time_s'] = unique_traj['calc_time_s'].cumsum()
        data['cumul_dist_km'] = unique_traj['calc_dist_km'].cumsum()
        data['cumul_pass_stops_n'] = unique_traj['pass_stops_n'].cumsum()
        data['cumul_time_s'] = data.cumul_time_s - unique_traj.cumul_time_s.transform('min')
        data['cumul_dist_km'] = data.cumul_dist_km - unique_traj.cumul_dist_km.transform('min')
        data['cumul_pass_stops_n'] = data.cumul_pass_stops_n - unique_traj.cumul_pass_stops_n.transform('min')
        # Save processed date
        num_pts_final = len(data)
        print(f"Kept {np.round(num_pts_final/num_pts_initial, 3)*100}% of original points")
        data.to_pickle(f"{kwargs['realtime_folder']}processed/{day}")
    print(f"PROCESSING COMPLETED: {kwargs['network_name']}")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    prepare_run(
        network_name="kcm",
        dates=data_utils.get_date_list("2023_03_15", 14),
        data_dropout=0.2,
        static_folder="./data/kcm_gtfs/",
        realtime_folder="./data/kcm_realtime/",
        timezone="America/Los_Angeles",
        epsg=32148,
        grid_bounds=[369903,37911,409618,87758],
        coord_ref_center=[386910,69022],
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )
    prepare_run(
        network_name="atb",
        dates=data_utils.get_date_list("2023_03_15", 14),
        data_dropout=0.2,
        static_folder="./data/atb_gtfs/",
        realtime_folder="./data/atb_realtime/",
        timezone="Europe/Oslo",
        epsg=32632,
        grid_bounds=[550869,7012847,579944,7039521],
        coord_ref_center=[569472,7034350],
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )