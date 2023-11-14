import datetime
import os
import pickle
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
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


def get_gtfs_shapes_lookup(gtfs_folder):
    """Get uniquely identified route shapes:trip_ids lookup from GTFS files."""
    routes = pd.read_csv(f"{gtfs_folder}routes.txt", low_memory=False, dtype=GTFS_LOOKUP)
    trips = pd.read_csv(f"{gtfs_folder}trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
    data = trips.merge(routes, on='route_id')
    data = data[['service_id','route_id','direction_id','trip_id']].drop_duplicates().sort_values(['service_id','route_id','direction_id','trip_id'])
    return data

def get_gtfs_shapes(gtfs_folder):
    """Use stop locations to create unique shapes from GTFS files."""
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
    """Join a set of GTFS files into a single dataframe. Each row is a trip + arrival time.
    Returns: pandas dataframe with merged GTFS data.
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
    """Make a copy of raw bus data with only a single operator."""
    files = os.listdir(old_folder)
    for file in files:
        print(f"Extracting {op_name} from {old_folder}{file} to {new_folder}{file}...")
        if file != ".DS_Store":
            data = pd.read_pickle(f"{old_folder}{file}")
            data = data[data[source_col]==op_name]
            pickle.dump(data, open(f"{new_folder}{file}", "wb"))


def extract_operator_gtfs(old_folder, new_folder, source_col_trips, source_col_stop_times, op_name):
    """First makes a copy of the GTFS directory, then overwrite the key files."""
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
    return res[:,1], res[:,2].astype(float), res[:,3]


def latest_available_static(day, static_folder):
    """Find the latest possible static data that is less than the realtime date."""
    static_available = [f for f in os.listdir(static_folder) if not f.startswith('.')]
    static_available = [datetime.datetime.strptime(x, "%Y_%m_%d") for x in static_available]
    static_needed = datetime.datetime.strptime(day, "%Y_%m_%d")
    static_possible = [sp for sp in static_available if sp < static_needed]
    static_best = max(static_possible)
    return datetime.datetime.strftime(static_best, "%Y_%m_%d")


def date_to_service_id(date_str, gtfs_folder):
    """Get a list of valid service ids for the given day of week and month."""
    calendar = pd.read_csv(f"{gtfs_folder}calendar.txt")
    weekdays = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday")
    obs_date = datetime.datetime.strptime(date_str, "%Y_%m_%d")
    obs_dow = weekdays[obs_date.weekday()]
    valid_service_ids = calendar[calendar[obs_dow]==1].copy()
    # Filter start/end
    valid_service_ids['service_start_dates'] = pd.to_datetime(valid_service_ids['start_date'], format='%Y%m%d')
    valid_service_ids['service_end_dates'] = pd.to_datetime(valid_service_ids['end_date'], format='%Y%m%d')
    valid_service_ids = valid_service_ids[valid_service_ids['service_start_dates']<=obs_date]
    valid_service_ids = valid_service_ids[valid_service_ids['service_end_dates']>=obs_date]
    return list(valid_service_ids.service_id)


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