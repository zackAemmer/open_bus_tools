import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString

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


def get_realtime_data(gtfs_folder):
    return None


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
        stop_dists, stop_idxs = spatial.get_closest_point(stop_coords[:,:2], trip_data[:,:2])
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