import logging
from pathlib import Path
import pickle

import lightning.pytorch as lp
import numpy as np
import pandas as pd

from openbustools import spatial, standardfeeds, trackcleaning
from openbustools.traveltime import grid


def process_data(**kwargs):
    logger.debug(f"PROCESSING: {kwargs['network_name']}")
    for day in kwargs['dates']:
        logger.debug(f"DAY: {day}")

        # Loading data and unifying column names/dtypes
        try:
            data = standardfeeds.load_standard_realtime(kwargs['realtime_folder'] / day)
        except:
            logger.warning(f"Could not load realtime data: {day}")
            continue
        logger.debug(f"Loaded and unified columns: {len(data):_} points")
        if len(data) == 0:
            continue
        else:
            initial_data_length = len(data)

        # Sensors seem to hold old positions right at start/end of trip
        data = trackcleaning.drop_track_ends(data, 'trip_id', 3)
        logger.debug(f"Removed trip start/end points: {len(data):_} points")
        if len(data) == 0:
            continue

        # Split full trip trajectories into smaller samples, resample
        data = trackcleaning.shingle(data, min_break=2, max_break=5)
        logger.debug(f"Shingled: {len(data):_} points")
        if len(data) == 0:
            continue

        # Project to local coordinate system, apply bounding box, center coords
        #TODO: Speed up?
        data = spatial.create_bounded_gdf(data, 'lon', 'lat', kwargs['epsg'], kwargs['coord_ref_center'], kwargs['grid_bounds'], kwargs['dem_file'])
        logger.debug(f"Created bounded gdf: {len(data):_} points")
        if len(data) == 0:
            continue

        # Calculate geometry features
        data['calc_dist_m'], data['calc_bear_d'], data['calc_time_s'] = spatial.calculate_gps_metrics(data, 'lon', 'lat', time_col='locationtime')
        logger.debug(f"Calculated point geometry: {len(data):_} points")
        if len(data) == 0:
            continue

        # Filter on attributes of individual points (may be invalid after recalculation)
        data = trackcleaning.filter_on_points(data, {'calc_dist_m': (0, 10_000), 'calc_time_s': (0, 5*60), 'elev_m': (-400, 20_000)})
        logger.debug(f"Filtered points: {len(data):_} points")
        if len(data) == 0:
            continue

        # Filter on attributes of full tracks (remove all invalid trajectories)
        data = trackcleaning.filter_on_tracks(data, {'calc_dist_m': (0, 10_000), 'calc_time_s': (0, 5*60), 'calc_speed_m_s': (0, 35)})
        logger.debug(f"Filtered tracks: {len(data):_} points")
        if len(data) == 0:
            continue

        # Add features based on time
        data = trackcleaning.add_time_features(data, kwargs['timezone'])
        logger.debug(f"Added time features: {len(data):_} points")
        if len(data) == 0:
            continue

        # Add features based on static feed
        best_static = standardfeeds.latest_available_static(day.split('.')[0], kwargs['static_folder'])
        data = trackcleaning.add_static_features(data, kwargs['static_folder'] / best_static, kwargs['epsg'])
        logger.debug(f"Added static features: {len(data):_} points")
        if len(data) == 0:
            continue

        # Add cumulative features
        data = trackcleaning.add_cumulative_features(data)
        logger.debug(f"Added cumulative features: {len(data):_} points")
        if len(data) == 0:
            continue

        # Calculate realtime grid features
        grid_bounds_xy, _ = spatial.project_bounds(kwargs['grid_bounds'], kwargs['coord_ref_center'], kwargs['epsg'])
        data_grid = grid.RealtimeGrid(grid_bounds_xy, 500)
        data_grid.build_cell_lookup(data[['locationtime','x','y','calc_speed_m_s','calc_bear_d']].copy())
        logger.debug(f"Calculated grid features: {len(data):_} points")
        if len(data) == 0:
            continue

        # Minimum training features for fast numpy memmap prediction on-disk
        data_id, data_n, data_c = trackcleaning.extract_training_features(data)
        data_g = data_grid.get_recent_points(data[['x','y','locationtime']].to_numpy(), 4).astype('int32')
        logger.debug(f"Extracted training features: {len(data):_} points")
        if len(data) == 0:
            continue

        ## Save processed data:
        # Full geodataframe for analysis
        processed_path = kwargs['realtime_folder'] / "processed" / "analysis"
        processed_path.mkdir(parents=True, exist_ok=True)
        data.to_pickle(processed_path / day)

        # Grid object for analysis
        processed_path = kwargs['realtime_folder'] / "processed" / "grid"
        processed_path.mkdir(parents=True, exist_ok=True)
        with open(processed_path / day, 'wb') as f:
            pickle.dump(data_grid, f)

        # Track training features
        processed_path = kwargs['realtime_folder'] / "processed" / "training" / day[:10]
        processed_path.mkdir(parents=True, exist_ok=True)
        np.save(processed_path / f"{day[:10]}_sid.npy", data_id)
        np.save(processed_path / f"{day[:10]}_n.npy", data_n)
        np.save(processed_path / f"{day[:10]}_c.npy", data_c)

        # Grid training features
        np.save(processed_path / f"{day[:10]}_g.npy", data_g)

        logger.debug(f"Final data: {len(data):_} points ({100*len(data)/initial_data_length:.2f}% of initial data)")
    logger.debug(f"PROCESSING COMPLETED: {kwargs['network_name']}\n")


if __name__=="__main__":
    lp.seed_everything(42, workers=True)
    logger = logging.getLogger('process_data')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    process_data(
        network_name="kcm",
        dates=standardfeeds.get_date_list("2023_03_15", 90),
        static_folder=Path("data/kcm_gtfs"),
        realtime_folder=Path("data/kcm_realtime"),
        dem_file=Path("data/kcm_spatial/usgs10m_dem_32148.tif"),
        timezone="America/Los_Angeles",
        epsg=32148,
        grid_bounds=[-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442],
        coord_ref_center=[-122.33761744472739, 47.61086041739939],
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )

    process_data(
        network_name="atb",
        dates=standardfeeds.get_date_list("2023_03_15", 90),
        static_folder=Path("data/atb_gtfs"),
        realtime_folder=Path("data/atb_realtime"),
        dem_file=Path("data/atb_spatial/eudtm30m_dem_32632.tif"),
        timezone="Europe/Oslo",
        epsg=32632,
        grid_bounds=[10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395],
        coord_ref_center=[10.392178466426625,63.430852975179626],
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )

    process_data(
        network_name="rut",
        dates=standardfeeds.get_date_list("2023_03_15", 90),
        static_folder=Path("data/rut_gtfs"),
        realtime_folder=Path("data/rut_realtime"),
        dem_file=Path("data/rut_spatial/eudtm30m_dem_32632.tif"),
        timezone="Europe/Oslo",
        epsg=32632,
        grid_bounds=[10.588056382271377,59.809956950105395,10.875078411359919,59.95982169587328],
        coord_ref_center=[10.742169939719487,59.911212837674746],
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
    )

    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        if standardfeeds.validate_realtime_data(row):
            logger.debug(f"{row['provider']}")
            process_data(
                network_name=row['uuid'],
                dates=standardfeeds.get_date_list("2024_01_03", 5),
                static_folder=Path('data', 'other_feeds', f"{row['uuid']}_static"),
                realtime_folder=Path('data', 'other_feeds', f"{row['uuid']}_realtime"),
                dem_file=[x for x in Path('data', 'other_feeds', f"{row['uuid']}_spatial").glob(f"*_{row['epsg_code']}.tif")][0],
                timezone=row['tz_str'],
                epsg=row['epsg_code'],
                grid_bounds=[row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']],
                coord_ref_center=[np.mean([row['min_lon'], row['max_lon']]), np.mean([row['min_lat'], row['max_lat']])],
                given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
            )