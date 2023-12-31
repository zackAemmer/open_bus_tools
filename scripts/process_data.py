import datetime
import logging
from pathlib import Path
import pickle
from zoneinfo import ZoneInfo

import geopandas as gpd
import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from openbustools import spatial, standardfeeds
from openbustools.traveltime import data_loader, grid


def prepare_run(**kwargs):
    """Pre-process training data and save to sub-folder."""
    print(f"PROCESSING: {kwargs['network_name']}")
    for day in kwargs['dates']:
        print(day)
        # Loading data and unifying column names/dtypes
        try:
            data = pd.read_pickle(Path(kwargs['realtime_folder'], day))
            num_pts_initial = len(data)
            print(f"Loaded {num_pts_initial:_} points")
        except:
            logging.warning(f"File failed to load: {day}")
            continue
        if num_pts_initial == 0:
            logging.warning(f"No points found: {day}")
            continue
        elif num_pts_initial < 100:
            logging.warning(f"Too few points found: {day}")
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

        # Split full trip trajectories into smaller samples, resample
        data = spatial.shingle(data, min_break=2, max_break=5, min_len=3, max_len=200)

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
        # First pt is dependent on prev trip metrics
        data = data.drop(data.groupby('shingle_id', as_index=False, sort=False).nth(0).index)
        data['calc_speed_m_s'] = data['calc_dist_m'] / data['calc_time_s']
        data['calc_dist_km'] = data['calc_dist_m'] / 1000.0

        # Filter trips with less than 2 points
        toss_ids = []
        pt_counts = data.groupby('shingle_id')[['calc_dist_m']].count()
        toss_ids.extend(list(pt_counts[pt_counts['calc_dist_m']<2].index))
        # Filter trips with outliers
        max_speed = 30
        secs_keep = 4 * 60
        toss_ids.extend(list(data[data['calc_speed_m_s']>max_speed]['shingle_id']))
        toss_ids.extend(list(data[data['calc_time_s']>secs_keep]['shingle_id']))
        toss_ids.extend(list(data[data['calc_dist_m']>secs_keep*max_speed]['shingle_id']))
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
        stop_times, stops, trips = standardfeeds.load_gtfs_files(Path(kwargs['static_folder'], best_static))
        static = stop_times.merge(stops, on='stop_id').sort_values(['trip_id','stop_sequence'])
        static = gpd.GeoDataFrame(static, geometry=gpd.points_from_xy(static.stop_lon, static.stop_lat), crs="EPSG:4326").to_crs(f"EPSG:{kwargs['epsg']}")
        # Filter any realtime trips that are not in the schedule
        data_filtered_static = data.drop(data[~data['trip_id'].isin(static.trip_id)].index)
        if len(data_filtered_static) > 0:
            data = data.drop(data[~data['trip_id'].isin(static.trip_id)].index)
            data['stop_id'], data['calc_stop_dist_m'], data['stop_sequence'] = standardfeeds.get_scheduled_arrival(data, static)
            data = data.merge(stop_times, on=['trip_id','stop_id','stop_sequence'], how='left')
            data = data.merge(trips, on='trip_id', how='left')
            data['calc_stop_dist_km'] = data['calc_stop_dist_m'] / 1000.0
        else:
            logging.warning(f"No data after joining static feed: {day}")
            data['stop_sequence'] = 0
            data['t_sch_sec_of_day'] = 0
            data['calc_stop_dist_m'] = 0
            data['calc_stop_dist_km'] = 0
            data['route_id'] = 'NotFound'
        # Passed stops
        data['pass_stops_n'] = data.groupby('shingle_id')['stop_sequence'].diff().fillna(0)
        # Scheduled time
        data['t_sec_of_day_start'] = data.groupby('shingle_id')[['t_sec_of_day']].transform('min')
        data['sch_time_s'] = data['t_sch_sec_of_day'] - data['t_sec_of_day_start']
        data['sch_time_s'] = data['sch_time_s'].fillna(0)

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
        grid_bounds_xy, _ = spatial.project_bounds(kwargs['grid_bounds'], kwargs['coord_ref_center'], kwargs['epsg'])
        data_grid = grid.RealtimeGrid(grid_bounds_xy, 500)
        data_grid.build_cell_lookup(data[['locationtime','x','y','calc_speed_m_s','calc_bear_d']].copy())

        # Save processed data
        num_pts_final = len(data)
        print(f"Kept {np.round(num_pts_final/num_pts_initial, 3)*100}% of original points")
        # Full geodataframe
        processed_path = Path(kwargs['realtime_folder'], "processed")
        processed_path.mkdir(parents=True, exist_ok=True)
        data.to_pickle(processed_path / day)
        # Grid object
        Path(processed_path / "grid").mkdir(parents=True, exist_ok=True)
        with open(Path(processed_path / "grid" / day), 'wb') as f:
            pickle.dump(data_grid, f)
        # Minimal training features
        data_id = data['shingle_id'].to_numpy().astype('int32')
        data_n = data[data_loader.NUM_FEAT_COLS].to_numpy().astype('int32')
        data_c = data[data_loader.MISC_CAT_FEATS].to_numpy().astype('S30')
        data_g = data_grid.get_recent_points(data[['x','y','locationtime']].to_numpy(), 4).astype('int32')
        with h5py.File(Path(processed_path / "samples.hdf5"), 'a') as f:
            if day in f.keys():
                del f[day]
            g = f.create_group(day)
            g.create_dataset('shingle_ids', data=data_id)
            g.create_dataset('feats_n', data=data_n)
            g.create_dataset('feats_c', data=data_c)
            g.create_dataset('feats_g', data=data_g)
    print(f"PROCESSING COMPLETED: {kwargs['network_name']}")


if __name__=="__main__":
    pl.seed_everything(42, workers=True)

    # prepare_run(
    #     network_name="kcm",
    #     dates=standardfeeds.get_date_list("2023_03_15", 38),
    #     static_folder="./data/kcm_gtfs/",
    #     realtime_folder="./data/kcm_realtime/",
    #     timezone="America/Los_Angeles",
    #     epsg=32148,
    #     # grid_bounds=[369903,37911,409618,87758],
    #     grid_bounds=[-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442],
    #     # coord_ref_center=[386910,69022],
    #     coord_ref_center=[-122.33761744472739, 47.61086041739939]
    #     dem_file="./data/kcm_spatial/usgs10m_dem_32148.tif",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    # )
    # prepare_run(
    #     network_name="atb",
    #     dates=standardfeeds.get_date_list("2023_03_15", 38),
    #     static_folder="./data/atb_gtfs/",
    #     realtime_folder="./data/atb_realtime/",
    #     timezone="Europe/Oslo",
    #     epsg=32632,
    #     # grid_bounds=[550869,7012847,579944,7039521],
    #     grid_bounds=[10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395],
    #     # coord_ref_center=[569472,7034350],
    #     coord_ref_center=[10.392178466426625,63.430852975179626],
    #     dem_file="./data/atb_spatial/eudtm30m_dem_32632.tif",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    # )
    # prepare_run(
    #     network_name="rut",
    #     dates=standardfeeds.get_date_list("2023_03_15", 38),
    #     static_folder="./data/rut_gtfs/",
    #     realtime_folder="./data/rut_realtime/",
    #     timezone="Europe/Oslo",
    #     epsg=32632,
    #     # grid_bounds=[589080,6631314,604705,6648420],
    #     grid_bounds=[10.588056382271377,59.809956950105395,10.875078411359919,59.95982169587328],
    #     # coord_ref_center=[597427,6642805],
    #     coord_ref_center=[10.742169939719487,59.911212837674746],
    #     dem_file="./data/rut_spatial/eudtm30m_dem_32632.tif",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    # )
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    cleaned_sources = cleaned_sources.iloc[35:]
    cleaned_sources = cleaned_sources[cleaned_sources['provider']=='York Region Transit']
    for i, row in cleaned_sources.iterrows():
        print(row['provider'])
        prepare_run(
            network_name=row['uuid'],
            dates=['2024_01_01.pkl'],
            static_folder=f"./data/other_feeds/{row['uuid']}_static/",
            realtime_folder=f"./data/other_feeds/{row['uuid']}_realtime/",
            timezone=row['tz_str'],
            epsg=row['epsg_code'],
            grid_bounds=[row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']],
            coord_ref_center=[np.mean([row['min_lon'], row['max_lon']]), np.mean([row['min_lat'], row['max_lat']])],
            dem_file=[x for x in Path('data', 'other_feeds', f"{row['uuid']}_spatial").glob(f"*_{row['epsg_code']}.tif")][0],
            given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
        )