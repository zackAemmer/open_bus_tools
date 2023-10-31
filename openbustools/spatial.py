import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def calculate_gps_metrics(gdf, time_col):
    """Calculate metrics between consecutive gps locations."""
    # Ensure no repeated time obs; can still have same time trip end/next start
    assert len(gdf.drop_duplicates(['trip_id', time_col])) == len(gdf)
    gdf_shifted = gdf.shift()
    consecutive_time_s = gdf[time_col] - gdf_shifted[time_col]
    consecutive_dist_m = gdf.distance(gdf_shifted, align=False)
    # The first row of each trip will overlap previous and should be removed
    return consecutive_time_s, consecutive_dist_m


def calculate_gps_dist(end_x, end_y, start_x, start_y):
    """Calculate the euclidean distance between a series of points."""
    x_diff = (end_x - start_x)
    y_diff = (end_y - start_y)
    dists = np.sqrt(x_diff**2 + y_diff**2)
    # Measured in degrees from the positive x axis
    # E==0, N==90, W==180, S==-90
    bearings = np.arctan2(y_diff, x_diff)*180/np.pi
    return dists, bearings


def calculate_trip_speeds(data):
    """Calculate speeds between consecutive trip locations."""
    x = data[['shingle_id','x','y','locationtime']]
    y = data[['shingle_id','x','y','locationtime']].shift()
    y.columns = [colname+"_shift" for colname in y.columns]
    z = pd.concat([x,y], axis=1)
    z['dist_diff'], z['bearing'] = calculate_gps_dist(z['x'].values, z['y'].values, z['x_shift'].values, z['y_shift'].values)
    z['time_diff'] = z['locationtime'] - z['locationtime_shift']
    z['speed_m_s'] = z['dist_diff'] / z['time_diff']
    return z['speed_m_s'].values, z['dist_diff'].values, z['time_diff'].values, z['bearing'].values


def get_closest_point(points, query_points):
    tree = KDTree(points)
    dists, idxs = tree.query(query_points)
    return dists, idxs


def get_adjacent_metric(shingle_group, adj_traces, d_buffer, t_buffer, b_buffer=None, orthogonal=False):
    """Calculate adjacent metric for each shingle from all other shingles in adj_traces."""
    # Set up spatial index for the traces
    tree = KDTree(adj_traces[:,:2])
    # Get time filter for the traces
    t_end = np.min(shingle_group[['locationtime']].values)
    t_start = t_end - t_buffer
    # Get the indices of adj_traces which fit dist buffer
    d_idxs = tree.query_ball_point(shingle_group[['x','y']].values, d_buffer)
    d_idxs = set([item for sublist in d_idxs for item in sublist])
    # Get the indices of adj_traces which fit time buffer
    t_idxs = (adj_traces[:,2] <= t_end) & (adj_traces[:,2] >= t_start)
    t_idxs = set(np.where(t_idxs)[0])
    # Get the indices of adj_traces which fit heading buffer
    if b_buffer is not None:
        if orthogonal == True:
            b_left = np.mean(shingle_group[['bearing']].values) + 90
            b_left_end = b_left + b_buffer
            b_left_start = b_left - b_buffer
            b_right = np.mean(shingle_group[['bearing']].values) - 90
            b_right_end = b_right + b_buffer
            b_right_start = b_right - b_buffer
            b_idxs = ((adj_traces[:,3] <= b_left_end) & (adj_traces[:,3] >= b_left_start)) | ((adj_traces[:,3] <= b_right_end) & (adj_traces[:,3] >= b_right_start))
        else:
            b_end = np.mean(shingle_group[['bearing']].values) + b_buffer
            b_start = np.mean(shingle_group[['bearing']].values) - b_buffer
            b_idxs = (adj_traces[:,3] <= b_end) & (adj_traces[:,3] >= b_start)
        b_idxs = set(np.where(b_idxs)[0])
        idxs = d_idxs & t_idxs & b_idxs
    else:
        idxs = d_idxs & t_idxs
    # Get the average speed of the trace and the relevant adj_traces
    target = np.mean(shingle_group[['speed_m_s']].values)
    if len(idxs) != 0:
        pred = np.mean(np.take(adj_traces[:,4], list(idxs), axis=0))
    else:
        pred = np.nan
    return (target, pred)


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