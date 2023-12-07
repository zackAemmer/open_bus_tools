from datetime import date, datetime, timedelta
import os
import pickle
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import scipy
from shapely.geometry import LineString

from openbustools import spatial

# Set of unified feature names and dtypes for variables in the GTFS-RT data
GTFSRT_NAMES = ['trip_id','file','locationtime','lat','lon','vehicle_id']
GTFSRT_TYPES = ['object','object','int','float','float','object']
GTFSRT_LOOKUP = dict(zip(GTFSRT_NAMES, GTFSRT_TYPES))

# Set of unified feature names and dtypes for variables in the GTFS data
GTFS_NAMES = ['trip_id','route_id','stop_id','stop_lat','stop_lon','arrival_time']
GTFS_TYPES = [str,str,str,float,float,str]
GTFS_LOOKUP = dict(zip(GTFS_NAMES, GTFS_TYPES))


def get_time_embeddings(timestamps, timezone='America/Los_Angeles'):
    """
    Converts a list of timestamps to time embeddings.

    Args:
        timestamps (list): A list of timestamps in seconds.
        timezone (str, optional): The timezone to convert the timestamps to. Defaults to 'America/Los_Angeles'.

    Returns:
        tuple: A tuple containing two arrays - minute_of_day and day_of_week.
            - minute_of_day: An array representing the minute of the day for each timestamp.
            - day_of_week: An array representing the day of the week for each timestamp.
    """
    local_times = pd.to_datetime(timestamps, unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
    minute_of_day = local_times.dt.hour * 60 + local_times.dt.minute
    day_of_week = local_times.dt.dayofweek
    return (minute_of_day.to_numpy(), day_of_week.to_numpy())


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
    data = data[['service_id','route_id','direction_id','trip_id']].drop_duplicates().sort_values(['service_id','route_id','direction_id','trip_id'])
    return data


def get_gtfs_shapes(gtfs_folder):
    """
    Use stop locations to create unique shapes from GTFS files.

    Args:
        gtfs_folder (str): The path to the folder containing the GTFS files.

    Returns:
        data (GeoDataFrame): A GeoDataFrame containing the unique shapes created from the GTFS files.
    """
    stops = pd.read_csv(f"{gtfs_folder}stops.txt", low_memory=False, dtype=GTFS_LOOKUP)
    stop_times = pd.read_csv(f"{gtfs_folder}stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
    data = stop_times.merge(stops, on='stop_id').sort_values(['trip_id','stop_sequence'])
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.stop_lon, data.stop_lat), crs="EPSG:4326")
    # Avoid making duplicate shapes; one per service/route/direction
    shapes_lookup = get_gtfs_shapes_lookup(gtfs_folder)
    data = data.merge(shapes_lookup, on='trip_id').drop_duplicates(['service_id','route_id','direction_id','stop_id']).sort_values(['service_id','route_id','direction_id','stop_sequence'])
    data = data.groupby(['service_id','route_id','direction_id'], as_index=False)['geometry'].apply(lambda x: LineString(x.tolist())).set_crs("EPSG:4326")
    return data


def merge_gtfs_files(gtfs_folder, epsg, coord_ref_center):
    """
    Join a set of GTFS files into a single dataframe. Each row is a trip + arrival time.

    Args:
        gtfs_folder (str): Path to the folder containing the GTFS files.
        epsg (int): EPSG code for the desired coordinate reference system.
        coord_ref_center (tuple): Tuple containing the coordinates of the reference center.

    Returns:
        pandas.DataFrame: Merged GTFS data.
    """
    z = pd.read_csv(gtfs_folder+"trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
    r = pd.read_csv(gtfs_folder+"routes.txt", low_memory=False, dtype=GTFS_LOOKUP)
    st = pd.read_csv(gtfs_folder+"stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
    sl = pd.read_csv(gtfs_folder+"stops.txt", low_memory=False, dtype=GTFS_LOOKUP)
    z = pd.merge(z,st,on="trip_id")
    z = pd.merge(z,sl,on="stop_id")
    z = pd.merge(z,r,on="route_id")
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


def extract_operator_gtfs(old_folder, new_folder, source_col_trips, source_col_stop_times, op_name):
    """
    Extracts the GTFS data for a specific operator from the old_folder and saves it to the new_folder.

    Args:
        old_folder (str): The path to the directory containing the original GTFS data.
        new_folder (str): The path to the directory where the extracted GTFS data will be saved.
        source_col_trips (str): The name of the column in the trips.txt file that contains the operator information.
        source_col_stop_times (str): The name of the column in the stop_times.txt file that contains the operator information.
        op_name (str): The name of the operator to extract.
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
    static_available = [f for f in os.listdir(static_folder) if not f.startswith('.')]
    static_available = [datetime.strptime(x, "%Y_%m_%d") for x in static_available]
    static_needed = datetime.strptime(day, "%Y_%m_%d")
    static_possible = [sp for sp in static_available if sp < static_needed]
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


def load_phone_data(phone_folder, window_len=120, polyorder=3):
    location = pd.read_csv(f"{phone_folder}Location.csv")
    location['time'] = location['time'].astype(str).str[:10].astype(int)
    location = location.add_prefix('location_')
    accelerometer = pd.read_csv(f"{phone_folder}Accelerometer.csv")
    accelerometer['y'] = accelerometer['y'] * -1
    accelerometer['time'] = accelerometer['time'].astype(str).str[:10].astype(int)
    # accelerometer['magnitude'] = np.linalg.norm(accelerometer[['z','y','x']].to_numpy(), ord=2, axis=1)
    accelerometer['y_smooth'] = scipy.signal.savgol_filter(accelerometer['y'].to_numpy(), window_length=window_len, polyorder=polyorder)
    accelerometer = accelerometer.add_prefix('accelerometer_')
    orientation = pd.read_csv(f"{phone_folder}Orientation.csv")
    orientation['time'] = orientation['time'].astype(str).str[:10].astype(int)
    orientation['pitch'] = orientation['pitch'] * 180 / np.pi
    orientation['pitch_smooth'] = scipy.signal.savgol_filter(orientation['pitch'].to_numpy(), window_length=window_len, polyorder=polyorder)
    orientation = orientation.add_prefix('orientation_')
    location = location.merge(accelerometer, left_on='location_time', right_on='accelerometer_time', how='left')
    location = location.merge(orientation, left_on='location_time', right_on='orientation_time', how='left')
    location = location.groupby('location_time').mean().reset_index()
    location['calc_time_s'] = np.diff(location['location_time'].to_numpy(), prepend=location['location_time'].to_numpy()[0])
    location['cumul_time_s'] = np.cumsum(location['calc_time_s'].to_numpy())
    location = location.dropna()
    return location


def filter_gtfs_w_phone(phone_df, gtfs_df, route_short_name, gtfs_calendar):
    """Filter bus schedule using best guess for trip_ids that correspond to a phone trip."""
    # Filter to the route number shown on the headsign
    filtered_df = gtfs_df[gtfs_df['route_short_name']==route_short_name]
    # # Filter by service ids that are active on day of week
    # weekdays = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday")
    # phone_start_time = datetime.fromtimestamp(int(str(min(phone_df.time))[:10]))
    # phone_start_time = phone_start_time.hour*3600 + phone_start_time.second
    # valid_service_ids = gtfs_calendar[gtfs_calendar[weekdays[phone_start_time.weekday()]]==1].service_id
    # filtered_df = filtered_df[filtered_df['service_id'].isin(valid_service_ids)]
    # Filter to direction 0; will need to be improved
    filtered_df = filtered_df[filtered_df['direction_id']==0]
    # # Filter to trips scheduled active when the phone started recording
    # trip_times = filtered_df[['trip_id','arrival_s']].groupby('trip_id', as_index=False).agg(['min','max'])
    # trip_times = trip_times[phone_start_time > trip_times[('arrival_s','min')]]
    # trip_times = trip_times[phone_start_time < trip_times[('arrival_s','max')]]
    # filtered_df = filtered_df[filtered_df['trip_id'].isin(trip_times[('trip_id','')])]
    # filtered_df = filtered_df.sort_values(['trip_id','arrival_s'])
    # Return only one trip_id in the dataframe for plotting, regardless of total remaining
    remaining_trip_ids = np.unique(filtered_df.trip_id)
    keep_trip_id = remaining_trip_ids[0]
    print(f"Filtered down to {len(remaining_trip_ids)} possible trip_ids; returning the first one ({keep_trip_id}).")
    filtered_df = filtered_df[filtered_df['trip_id']==keep_trip_id]
    return filtered_df.copy(), remaining_trip_ids