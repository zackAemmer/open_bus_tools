import datetime
import pickle
from zoneinfo import ZoneInfo

import geopandas as gpd
import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from openbustools import spatial, standardfeeds
from openbustools.traveltime import data_loader, grid


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
        if num_pts_initial == 0:
            continue
        data['file'] = day[:10]
        data = data[kwargs['given_names']]
        data.columns = standardfeeds.GTFSRT_LOOKUP.keys()
        data['locationtime'] = data['locationtime'].astype(float)
        data = data.astype(standardfeeds.GTFSRT_LOOKUP)

        # Sensors seem to hold old positions right at start/end of trip
        data = data.reset_index(drop=True)
        for _ in range(3):
            data = data.drop(data.groupby('trip_id', as_index=False).nth(0).index)
            data = data.drop(data.groupby('trip_id', as_index=False).nth(-1).index)

        # # Avoid sensors recording at regular intervals
        # drop_indices = np.random.choice(data.index, int(kwargs['data_dropout']*len(data)), replace=False)
        # data = data[~data.index.isin(drop_indices)].reset_index(drop=True)

        # Split full trip trajectories into smaller samples, resample
        data = spatial.shingle(data, 2, 5, 3, 60)

        # Project to local coordinate system, apply bounding box, center coords
        data = spatial.create_bounded_gdf(data, 'lon', 'lat', kwargs['epsg'], kwargs['coord_ref_center'], kwargs['grid_bounds'], kwargs['dem_file'])

        # Calculate geometry features
        data['calc_time_s'] = data['locationtime'] - data['locationtime'].shift(1)
        data['calc_dist_m'], data['calc_bear_d'] = spatial.calculate_gps_metrics(data, 'lon', 'lat')
        # If time between points is too long, or distance is too short, or not found in DEM, drop
        data = data[(data['calc_dist_m']>0) & (data['calc_time_s']>=1) & (data['elev_m']>-400)].copy()
        # Re-calculate geometry features w/o missing points
        data['calc_time_s'] = data['locationtime'] - data['locationtime'].shift(1)
        data['calc_dist_m'], data['calc_bear_d'] = spatial.calculate_gps_metrics(data, 'lon', 'lat')
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
        # Filter the list of full shingles w/invalid points
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
        stop_times['t_sch_hour'] = stop_times['arrival_time'].str.split(':').str[0].astype(int)
        stop_times['t_sch_min'] = stop_times['arrival_time'].str.split(':').str[1].astype(int)
        stop_times['t_sch_sec'] = stop_times['arrival_time'].str.split(':').str[2].astype(int)
        stop_times['t_sch_min_of_day'] = (stop_times['t_sch_hour']*60) + stop_times['t_sch_min']
        stop_times['t_sch_sec_of_day'] = (stop_times['t_sch_hour']*60*60) + (stop_times['t_sch_min']*60) + stop_times['t_sch_sec']
        stop_times = stop_times.sort_values(['trip_id','t_sch_sec_of_day'])
        stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount()
        static = stop_times.merge(stops, on='stop_id').sort_values(['trip_id','stop_sequence'])
        static = gpd.GeoDataFrame(static, geometry=gpd.points_from_xy(static.stop_lon, static.stop_lat), crs="EPSG:4326").to_crs(f"EPSG:{kwargs['epsg']}")
        # Filter any realtime trips that are not in the schedule
        data_filter_static = data.drop(data[~data['trip_id'].isin(static.trip_id)].index)
        if len(data_filter_static) > 0:
            data = data.drop(data[~data['trip_id'].isin(static.trip_id)].index)
            data['stop_id'], data['calc_stop_dist_m'], data['stop_sequence'] = standardfeeds.get_scheduled_arrival(data, static)
            data = data.merge(stop_times, on=['trip_id','stop_id','stop_sequence'], how='left')
            data = data.merge(trips, on='trip_id', how='left')
            data['calc_stop_dist_km'] = data['calc_stop_dist_m'] / 1000.0
        else:
            continue
            data['stop_id'] = np.nan
            data['calc_stop_dist_m'] = np.nan
            data['calc_stop_dist_km'] = np.nan
            data['stop_sequence'] = np.nan
            data['route_id'] = np.nan
            data['t_sch_sec_of_day'] = np.nan
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
        data['cumul_dist_m'] = data['cumul_dist_km'] * 1000
        data['cumul_pass_stops_n'] = unique_traj['pass_stops_n'].cumsum()
        data['cumul_time_s'] = data.cumul_time_s - unique_traj.cumul_time_s.transform('min')
        data['cumul_dist_km'] = data.cumul_dist_km - unique_traj.cumul_dist_km.transform('min')
        data['cumul_pass_stops_n'] = data.cumul_pass_stops_n - unique_traj.cumul_pass_stops_n.transform('min')
        data['data_folder'] = kwargs['realtime_folder']

        # Realtime grid features
        data_grid = grid.RealtimeGrid(kwargs['grid_bounds'], 500)
        data_grid.build_cell_lookup(data[['locationtime','x','y','calc_speed_m_s','calc_bear_d']].copy())

        # Save processed data
        num_pts_final = len(data)
        print(f"Kept {np.round(num_pts_final/num_pts_initial, 3)*100}% of original points")
        # Full geodataframe
        data.to_pickle(f"{kwargs['realtime_folder']}processed/{day}")
        # Grid object
        with open(f"{kwargs['realtime_folder']}processed/grid/{day}", 'wb') as f:
            pickle.dump(data_grid, f)
        # Minimal training features
        data_id = data['shingle_id'].to_numpy().astype('int32')
        data_n = data[data_loader.NUM_FEAT_COLS].to_numpy().astype('int32')
        data_c = data[data_loader.MISC_CAT_FEATS].to_numpy().astype('S10')
        data_g = data_grid.get_recent_points(data[['x','y','locationtime']].to_numpy(), 4).astype('int32')
        with h5py.File(f"{kwargs['realtime_folder']}processed/samples.hdf5", 'a') as f:
            if day in f.keys():
                del f[day]
            g = f.create_group(day)
            g.create_dataset('shingle_ids', data=data_id)
            g.create_dataset('feats_n', data=data_n)
            g.create_dataset('feats_c', data=data_c)
            g.create_dataset('feats_g', data=data_g)
    print(f"PROCESSING COMPLETED: {kwargs['network_name']}")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    prepare_run(
        network_name="kcm",
        dates=standardfeeds.get_date_list("2023_03_15", 37),
        # data_dropout=0.2,
        static_folder="./data/kcm_gtfs/",
        realtime_folder="./data/kcm_realtime/",
        timezone="America/Los_Angeles",
        epsg=32148,
        grid_bounds=[369903,37911,409618,87758],
        coord_ref_center=[386910,69022],
        dem_file="./data/kcm_spatial/usgs10m_dem_32148.tif",
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )
    prepare_run(
        network_name="atb",
        dates=standardfeeds.get_date_list("2023_03_15", 37),
        # data_dropout=0.2,
        static_folder="./data/atb_gtfs/",
        realtime_folder="./data/atb_realtime/",
        timezone="Europe/Oslo",
        epsg=32632,
        grid_bounds=[550869,7012847,579944,7039521],
        coord_ref_center=[569472,7034350],
        dem_file="./data/atb_spatial/eudtm30m_dem_32632.tif",
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )
    # prepare_run(
    #     network_name="rut",
    #     dates=standardfeeds.get_date_list("2023_03_15", 90),
    #     data_dropout=0.2,
    #     static_folder="./data/rut_gtfs/",
    #     realtime_folder="./data/rut_realtime/",
    #     timezone="Europe/Oslo",
    #     epsg=32632,
    #     grid_bounds=[589080,6631314,604705,6648420],
    #     coord_ref_center=[597427,6642805],
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    # )