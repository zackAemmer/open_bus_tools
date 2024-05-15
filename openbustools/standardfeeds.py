from datetime import date, datetime, timedelta
import os
from pathlib import Path
import pickle
import shutil

import geopandas as gpd
import gtfs_kit as gk
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

from openbustools import spatial
from openbustools.drivecycle import trajectory


# Set of unified feature names and dtypes for variables in the GTFS data
GTFS_NAMES = ['trip_id','route_id','stop_id','stop_lat','stop_lon','arrival_time']
GTFS_TYPES = ['object','object','object','float','float','object']
GTFS_LOOKUP = dict(zip(GTFS_NAMES, GTFS_TYPES))
# Set of unified feature names and dtypes for variables in the GTFS-RT data
GTFSRT_NAMES = ['trip_id','locationtime','lat','lon','vehicle_id']
GTFSRT_TYPES = ['object','int','float','float','object']
GTFSRT_LOOKUP = dict(zip(GTFSRT_NAMES, GTFSRT_TYPES))


def load_standard_static(folderpath):
    """
    Join a set of GTFS files into a single dataframe. Each row is a trip + arrival time.

    Args:
        folderpath (str): Path to the folder containing the unzipped static files.

    Returns:
        tuple: A tuple containing three dataframes - stop_times, stops, and trips.
    """
    # Read stop locations, trips/times, and route ids
    stop_times = pd.read_csv(Path(folderpath, "stop_times.txt"), low_memory=False, dtype=GTFS_LOOKUP)[['trip_id','stop_id','arrival_time','stop_sequence']].dropna()
    stops = pd.read_csv(Path(folderpath, "stops.txt"), low_memory=False, dtype=GTFS_LOOKUP)[['stop_id','stop_lon','stop_lat']].sort_values('stop_id').dropna()
    trips = pd.read_csv(Path(folderpath, "trips.txt"), low_memory=False, dtype=GTFS_LOOKUP)[['trip_id','service_id','route_id']].sort_values(['service_id','route_id','trip_id']).dropna()
    return (stop_times, stops, trips)


def load_standard_realtime(filepath):
    """
    Load standard realtime data from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        DataFrame: The loaded real-time data.

    """
    data = pd.read_pickle(filepath)
    data = data[GTFSRT_NAMES]
    data.columns = GTFSRT_LOOKUP.keys()
    data['locationtime'] = data['locationtime'].astype(float)
    data = data.astype(GTFSRT_LOOKUP)
    data['realtime_filename'] = filepath.name
    data['realtime_foldername'] = str(filepath.parent)
    data = data.reset_index(drop=True)
    return data


def combine_static_stops(stop_times, stops, epsg):
    """
    Combines stop_times and stops dataframes to create a GeoDataFrame with static stop information.

    Args:
        stop_times (DataFrame): DataFrame containing stop times information.
        stops (DataFrame): DataFrame containing stop information.
        epsg (int): EPSG code for the desired coordinate reference system.

    Returns:
        static (GeoDataFrame): GeoDataFrame with combined static stop information.
    """
    # Deal with schedule crossing midnight, get scheduled arrival
    stop_times['t_sch_sec_of_day'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in stop_times['arrival_time'].str.split(":")]
    stop_times = stop_times.sort_values(['trip_id','t_sch_sec_of_day'])
    stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount()
    static = stop_times.merge(stops, on='stop_id').sort_values(['trip_id','stop_sequence'])
    static = gpd.GeoDataFrame(static, geometry=gpd.points_from_xy(static.stop_lon, static.stop_lat), crs="EPSG:4326").to_crs(f"EPSG:{epsg}")
    return static


def validate_realtime_data(row):
    """
    Validates the realtime data for a given city.

    Args:
        row (pandas.Series): The row containing the city info to be validated.

    Returns:
        bool: True if there is loadable realtime data, False otherwise.
    """
    uuid = row['uuid']
    provider_path = Path('data', 'other_feeds', f"{uuid}_realtime")
    available_files = [x.name for x in provider_path.glob('*.pkl')]
    if len(available_files) != 0:
        for file in available_files:
            data = pd.read_pickle(provider_path / file)
            if len(data) == 0:
                return False
        return True
    else:
        return False


def get_date_list(start, n_days):
    """
    Get a list of date strings starting at a given day and continuing for n days.

    Args:
        start (str): Date string formatted as 'yyyy_mm_dd'.
        n_days (int): Number of days forward to include from start day.

    Returns:
        list: List of date strings in the format 'yyyy_mm_dd.pkl'.
    """
    year, month, day = start.split("_")
    base = date(int(year), int(month), int(day))
    date_list = [base + timedelta(days=x) for x in range(n_days)]
    return [f"{date.strftime('%Y_%m_%d')}.pkl" for date in date_list]


def get_trip_start_and_end_times(static_feed):
    stop_times = static_feed.get_stop_times()
    stop_times = stop_times.sort_values(['trip_id','stop_sequence'])[['trip_id','departure_time']]
    stop_times_first = stop_times.groupby('trip_id').first()
    stop_times_last = stop_times.groupby('trip_id').last()
    stop_times = stop_times_first.merge(stop_times_last, on='trip_id', suffixes=('_first','_last'))
    stop_times = stop_times.reset_index()
    stop_times['t_hour_first'] = stop_times['departure_time_first'].str.split(':').str[0].astype(int)
    stop_times['t_min_first'] = stop_times['departure_time_first'].str.split(':').str[1].astype(int)
    stop_times['t_min_of_day_first'] = stop_times['t_hour_first']*60 + stop_times['t_min_first']
    stop_times['t_hour_last'] = stop_times['departure_time_last'].str.split(':').str[0].astype(int)
    stop_times['t_min_last'] = stop_times['departure_time_last'].str.split(':').str[1].astype(int)
    stop_times['t_min_of_day_last'] = stop_times['t_hour_last']*60 + stop_times['t_min_last']
    stop_times = stop_times[['trip_id','t_min_of_day_first','t_min_of_day_last']].copy()
    return stop_times


def get_gtfs_shapes_lookup(gtfs_folder):
    """
    Get uniquely identified route shapes:trip_ids lookup from GTFS files.

    Args:
        gtfs_folder (str): The path to the folder containing the GTFS files.

    Returns:
        data (DataFrame): A DataFrame containing the uniquely identified route shapes:trip_ids lookup.
    """
    routes = pd.read_csv(f"{gtfs_folder}routes.txt", low_memory=False, dtype=GTFS_LOOKUP)
    trips = pd.read_csv(f"{gtfs_folder}trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
    data = trips.merge(routes, on='route_id')
    data = data[['service_id','route_id','trip_id']].drop_duplicates().sort_values(['service_id','route_id','trip_id'])
    return data


def get_gtfs_shapes(gtfs_folder, epsg=None, stop_dist_filter=None):
    """
    Use stop locations to create unique shapes for each trip in GTFS files.

    Args:
        gtfs_folder (str): The path to the folder containing the GTFS files.
        epsg (int, optional): The EPSG code for the desired coordinate reference system. Defaults to None.
        stop_dist_filter (float, optional): The maximum distance between consecutive stops to consider when filtering shapes. Defaults to None.

    Returns:
        data (GeoDataFrame): A GeoDataFrame containing the unique shapes created from the GTFS files.
    """
    stop_times, stops, trips = load_standard_static(gtfs_folder)
    data = stop_times.merge(stops, on='stop_id').sort_values(['trip_id','stop_sequence'])
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.stop_lon, data.stop_lat), crs="EPSG:4326").to_crs(f"EPSG: {epsg}")
    if stop_dist_filter:
        # Filter first point of trip
        data['calc_dist_m'] = shapely.distance(data.geometry, data.geometry.shift())
        data = data.reset_index()
        drop_ids = data.groupby('trip_id').nth(0).index
        data = data.drop(index=drop_ids)
        # Filter trips with too-far stop distances
        drop_ids = data[data['trip_id'].isin(data[data['calc_dist_m']>=stop_dist_filter].trip_id)].index
        data = data.drop(index=drop_ids)
        # Filter trips with too-few stops
        data.groupby(['trip_id'], as_index=False).count()
        counts = data.groupby('trip_id').count()['geometry']
        drop_ids = counts[counts<5].index
        data = data[~data['trip_id'].isin(drop_ids)]
    # Avoid making duplicate shapes; one per service/route/direction
    shapes_lookup = get_gtfs_shapes_lookup(gtfs_folder)
    data = pd.merge(shapes_lookup, data, on=['trip_id'])[['service_id','route_id','trip_id','stop_id','stop_sequence','geometry']].sort_values(['service_id','route_id','trip_id','stop_sequence']).drop_duplicates()
    data = data.groupby(['service_id','route_id','trip_id'], as_index=False)['geometry'].apply(lambda x: LineString(x.tolist()))
    data = gpd.GeoDataFrame(data, geometry='geometry', crs=f"EPSG: {epsg}")
    return data


def extract_operator(old_folder, new_folder, source_col, op_name):
    """
    Save a copy of raw bus data with only a single operator.

    Args:
        old_folder (str): The path to the folder containing the raw bus data.
        new_folder (str): The path to the folder where the extracted data will be saved.
        source_col (str): The name of the column in the data that contains the operator information.
        op_name (str): The name of the operator to extract.
    """
    files = os.listdir(old_folder)
    for file in files:
        print(f"Extracting {op_name} from {old_folder}{file} to {new_folder}{file}...")
        if file != ".DS_Store":
            data = pd.read_pickle(f"{old_folder}{file}")
            data = data[data[source_col]==op_name]
            pickle.dump(data, open(f"{new_folder}{file}", "wb"))


def extract_operator_gtfs(old_folder, new_folder, area):
    gtfs_folders = [x for x in list(Path(old_folder).glob('*')) if x.is_dir()]
    for gtfs_date in gtfs_folders:
        print(f"Extracting GTFS for {gtfs_date}...")
        feed = gk.read_feed(gtfs_date, dist_units='km')
        feed_atb = feed.restrict_to_area(area)
        feed_atb.write(new_folder / gtfs_date.name)


def get_scheduled_arrival(realtime, static):
    """Find nearest appropriate static stop for a set of realtime coordinates. 
    The point is to use spatial tree to only get distance to nearest 
    neighbor for each trip, rather than all possible neighbors which requires 
    cross join.

    Args:
        realtime (DataFrame): DataFrame containing realtime data with trip_id, coordinates, and order.
        static (DataFrame): DataFrame containing static data with trip_id, stop_id, stop_sequence, and coordinates.

    Returns:
        tuple: A tuple containing three arrays - stop_ids, stop_distances, and stop_sequences.
            - stop_ids (array): Array of stop_ids corresponding to the nearest stops for each realtime point.
            - stop_distances (array): Array of distances from each realtime point to its nearest stop.
            - stop_sequences (array): Array of stop sequences corresponding to the nearest stops for each realtime point.
    """
    # Pull id and coordinate data from dataframes
    realtime_trips = realtime.trip_id.values
    realtime_order = np.arange(len(realtime))
    realtime_coords = np.column_stack([realtime.geometry.x.values, realtime.geometry.y.values])
    static_trips = static.trip_id.values
    static_stops = static.stop_id.values
    static_stop_seqs = static.stop_sequence.values
    static_coords = np.column_stack([static.geometry.x.values, static.geometry.y.values])
    # Create dictionary mapping trip_ids to lists of stop coordinates
    id_to_stops = {}
    for i, tripid in enumerate(static_trips):
        # If the key does not exist, insert the second argument. Otherwise return the value. Append afterward regardless.
        id_to_stops.setdefault(tripid,[]).append((static_coords[i], static_stops[i], static_stop_seqs[i]))
    # Repeat for realtime trips
    id_to_data = {}
    for i, tripid in enumerate(realtime_trips):
        id_to_data.setdefault(tripid,[]).append((realtime_coords[i], realtime_order[i]))
    # Iterate over each unique trip, getting closest stops for all points from that trip
    res_counter = 0
    res = np.zeros((len(realtime),4), dtype=object)
    for key, value in id_to_data.items():
        realtime_pts = np.vstack([x[0] for x in value])
        realtime_order_i = [x[1] for x in value]
        stop_pts = np.vstack([x[0] for x in id_to_stops[key]])
        all_stop_ids = [x[1] for x in id_to_stops[key]]
        all_stop_seqs = [x[2] for x in id_to_stops[key]]
        stop_dists, stop_idxs = spatial.get_point_distances(stop_pts, realtime_pts)
        stop_ids = np.array([all_stop_ids[i] for i in stop_idxs], dtype='object')
        stop_seqs = np.array([all_stop_seqs[i] for i in stop_idxs], dtype='object')
        # Return for each point; stop id, distance to stop
        res[res_counter:res_counter+len(stop_idxs), 0] = realtime_order_i
        res[res_counter:res_counter+len(stop_idxs), 1] = stop_ids
        res[res_counter:res_counter+len(stop_idxs), 2] = stop_dists
        res[res_counter:res_counter+len(stop_idxs), 3] = stop_seqs
        res_counter += len(stop_idxs)
    # Sort the data points from aggregated trips back into their respective shingles
    original_order = np.argsort(res[:,0])
    res = res[original_order,:]
    return (res[:,1], res[:,2].astype(float), res[:,3])


def latest_available_static(day, static_folder):
    """
    Find the latest possible static data that is less than the realtime date.

    Args:
        day (str): The date in the format "%Y_%m_%d".
        static_folder (str): The path to the folder containing the static data.

    Returns:
        str: The latest available static data date in the format "%Y_%m_%d".
    """
    # Use path glob to get only the directories, not zip files in the static folder
    static_available = [x.name for x in Path(static_folder).glob("*/") if x.is_dir()]
    static_available = [datetime.strptime(x, "%Y_%m_%d") for x in static_available]
    static_needed = datetime.strptime(day, "%Y_%m_%d")
    static_possible = [sp for sp in static_available if sp < static_needed]
    # If there are no static before the needed date, use the earliest available
    if len(static_possible) == 0:
        static_best = min(static_available)
    else:
        static_best = max(static_possible)
    return datetime.strftime(static_best, "%Y_%m_%d")


def date_to_service_id(date_str, gtfs_folder):
    """
    Get a list of valid service ids for the given day of week and month.

    Args:
        date_str (str): The date string in the format 'YYYY_MM_DD'.
        gtfs_folder (str): The path to the GTFS folder.

    Returns:
        list: A list of valid service ids for the given date.
    """
    calendar = pd.read_csv(f"{gtfs_folder}calendar.txt")
    weekdays = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday")
    obs_date = datetime.strptime(date_str, "%Y_%m_%d")
    obs_dow = weekdays[obs_date.weekday()]
    valid_service_ids = calendar[calendar[obs_dow]==1].copy()
    # Filter start/end
    valid_service_ids['service_start_dates'] = pd.to_datetime(valid_service_ids['start_date'], format='%Y%m%d')
    valid_service_ids['service_end_dates'] = pd.to_datetime(valid_service_ids['end_date'], format='%Y%m%d')
    valid_service_ids = valid_service_ids[valid_service_ids['service_start_dates']<=obs_date]
    valid_service_ids = valid_service_ids[valid_service_ids['service_end_dates']>=obs_date]
    return list(valid_service_ids.service_id)


def segmentize_shapes(static_feed, epsg, point_sep_m=300):
    shape_geometries = static_feed.geometrize_shapes().copy().set_crs(4326).to_crs(epsg)
    # Get regularly spaced points on each shape in the static feed
    distances = [np.arange(0,line.length,point_sep_m) for line in shape_geometries['geometry']]
    shape_ids = np.repeat(shape_geometries['shape_id'], [len(x) for x in distances])
    seq_ids = [np.arange(len(x)) for x in distances]
    points = [line.interpolate(d) for line, d in zip(shape_geometries['geometry'], distances)]
    route_shape_points = gpd.GeoDataFrame(geometry=np.concatenate(points), crs=epsg)
    route_shape_points['shape_id'] = shape_ids.values
    route_shape_points['seq_id'] = np.concatenate(seq_ids)
    route_shape_points['dist_shape_m'] = np.concatenate(distances)
    route_shape_points = {k:d for k, d in route_shape_points.groupby("shape_id")}
    return route_shape_points


def get_gnss_trip(gnss_solution_file, start_epoch, end_epoch, time_offset=18):
    data = pd.read_csv(gnss_solution_file, sep='\s+', header=None)
    data.columns = ["day", "time", "lat", "lon", "elev_m", "Q", "ns", "sdn", "sde", "sdu", "sdne", "sdeu", "sdun", "age", "ratio"]
    data['locationtime'] = (pd.to_datetime(data.day.astype(str) + " " + data.time.astype(str)).astype(int) * 1e-9).astype(int) - time_offset
    data = data[data['locationtime']>=start_epoch]
    data = data[data['locationtime']<=end_epoch]
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs="EPSG:4326")
    data = data.groupby('locationtime').first().reset_index()
    return data


def get_realtime_trip(realtime_folder, day, veh_id, start_epoch, end_epoch, veh_id_col='vehicleid'):
    # Load corresponding realtime data
    file_path = Path(realtime_folder, f"{day}.pkl")
    data = pd.read_pickle(file_path)
    data = data[data[veh_id_col].astype(int).astype(str)==veh_id]
    data = data[data['locationtime'].astype(int)>=start_epoch]
    data = data[data['locationtime'].astype(int)<=end_epoch]
    return data


def get_phone_trip(phone_folder, timezone, chop_n=None, time_offset=0):
    """
    Combines sensor data from different files into a single dataframe.

    Args:
        phone_folder (str): The path to the folder containing the sensor data files.
        timezone (str): The timezone of the data.
        chop_n (int, optional): The number of rows to remove from the beginning and end of the data. Defaults to None.
        time_offset (int, optional): The time offset to add to the location data to match GPS time. Defaults to 19.

    Returns:
        pandas.DataFrame: A dataframe containing the combined sensor data.
    """
    # Location; defines time index start
    location = pd.read_csv(Path(phone_folder, "Location.csv"))[['time','longitude','latitude','altitudeAboveMeanSeaLevel','bearing','speed']]
    location['locationtime'] = (location['time'] * 1e-9).astype(int) + time_offset
    # Accelerometer
    accelerometer = pd.read_csv(Path(phone_folder, "Accelerometer.csv"))[['time','y']]
    accelerometer['locationtime'] = (accelerometer['time'] * 1e-9).astype(int) + time_offset
    # Orientation
    orientation = pd.read_csv(Path(phone_folder, "Orientation.csv"))[['time','pitch']]
    orientation['locationtime'] = (orientation['time'] * 1e-9).astype(int) + time_offset
    orientation['pitch'] = orientation['pitch'] * 180 / np.pi
    # Join dataframes
    all_sensors = location.merge(accelerometer, on='locationtime', how='left')
    all_sensors = all_sensors.merge(orientation, on='locationtime', how='left')
    all_sensors = all_sensors.ffill().bfill()
    all_sensors = all_sensors.groupby('locationtime', as_index=False).mean()
    if chop_n:
        all_sensors = all_sensors.iloc[chop_n:-chop_n,:]
    all_sensors['cumul_time_s'] = all_sensors['locationtime'] - all_sensors['locationtime'].min()
    # Metadata
    annotation = pd.read_csv(Path(phone_folder, "Annotation.csv"))
    veh_id, short_name = str(phone_folder).split("/")[-1].split("-")[0:2] 
    start_timestamp = pd.to_datetime(int(all_sensors.iloc[0].locationtime), unit='s').tz_localize('UTC').tz_convert(timezone)
    t_day = start_timestamp.strftime('%Y_%m_%d')
    t_min_of_day = int(start_timestamp.strftime('%H:%M:%S').split(":")[0])*60 + int(start_timestamp.strftime('%H:%M:%S').split(":")[1])
    t_day_of_week = start_timestamp.strftime('%w')
    start_epoch = int(all_sensors.iloc[0].locationtime)
    end_epoch = int(all_sensors.iloc[-1].locationtime)
    metadata = {
        'folder': phone_folder,
        'short_name': short_name,
        'veh_id': veh_id,
        't_day': t_day,
        't_min_of_day': t_min_of_day,
        't_day_of_week': t_day_of_week,
        'start_epoch': start_epoch,
        'end_epoch': end_epoch,
        'timezone': timezone,
        'chop_n': chop_n,
        'time_offset': time_offset,
    }
    return (metadata, all_sensors)


def get_gnss_trajectory(phone_traj, gnss_solution_file, resample=False):
    metadata_phone = phone_traj.traj_attr
    data_gnss = get_gnss_trip(gnss_solution_file, metadata_phone['start_epoch'], metadata_phone['end_epoch'])
    # Adjust phone trajectory metadata for the realtime
    metadata_gnss = metadata_phone.copy()
    metadata_gnss.update({
        "start_epoch_gnss": data_gnss['locationtime'].iloc[0].astype(int),
        "end_epoch_gnss": data_gnss['locationtime'].iloc[-1].astype(int),
    })
    # Create trajectory
    gnss_traj = trajectory.Trajectory(
        point_attr={
            "lon": data_gnss.lon.to_numpy(),
            "lat": data_gnss.lat.to_numpy(),
            "locationtime": data_gnss.locationtime.to_numpy(),
            "measured_elev_m": data_gnss.elev_m.to_numpy(),
        },
        traj_attr=metadata_gnss,
        resample=resample
    )
    return gnss_traj


def get_realtime_trajectory(phone_traj, realtime_folder, resample=False):
    metadata_phone = phone_traj.traj_attr
    data_realtime = get_realtime_trip(realtime_folder, metadata_phone['t_day'], metadata_phone['veh_id'], metadata_phone['start_epoch'], metadata_phone['end_epoch'], veh_id_col='vehicle_id')
    # Adjust phone trajectory metadata for the realtime
    metadata_realtime = metadata_phone.copy()
    metadata_realtime.update({
        "start_epoch_realtime": data_realtime['locationtime'].iloc[0].astype(int),
        "end_epoch_realtime": data_realtime['locationtime'].iloc[-1].astype(int),
    })
    # Create trajectory
    realtime_traj = trajectory.Trajectory(
        point_attr={
            "lon": data_realtime.lon.to_numpy(),
            "lat": data_realtime.lat.to_numpy(),
            "locationtime": data_realtime.locationtime.to_numpy(),
        },
        traj_attr=metadata_realtime,
        resample=resample
    )
    return realtime_traj


def get_phone_trajectory(phone_trajectory_folder, timezone, epsg, coord_ref_center, dem_file, chop_n=None, resample=False):
    metadata_phone, data_phone = get_phone_trip(phone_trajectory_folder, timezone, chop_n)
    # Add args to trajectory metadata
    metadata_phone['coord_ref_center'] = coord_ref_center
    metadata_phone['epsg'] = epsg
    metadata_phone['dem_file'] = dem_file
    # Create trajectory
    phone_traj = trajectory.Trajectory(
        point_attr={
            "lon": data_phone.longitude.to_numpy(),
            "lat": data_phone.latitude.to_numpy(),
            "locationtime": data_phone.locationtime.to_numpy(),
            "measured_elev_m": data_phone.altitudeAboveMeanSeaLevel.to_numpy(),
            "measured_speed_m_s": data_phone.speed.to_numpy(),
            "measured_bear_d": data_phone.bearing.to_numpy(),
            "measured_pitch_d": data_phone.pitch.to_numpy(),
        },
        traj_attr=metadata_phone,
        resample=resample
    )
    return phone_traj


def clean_parametrix(data_path):
    df = pd.read_csv(data_path)
    df = df.melt(id_vars=['DateTime'], var_name='vehicle_id_metric', value_name='value')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['vehicle_id'] = df['vehicle_id_metric'].str.split(' - ').str[0]
    df['metric'] = df['vehicle_id_metric'].str.split(' - ').str[1]
    df = df.drop(columns=['vehicle_id_metric'])
    return df