import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def calculate_speed(gdf, time_col):
    """Calculate speed between consecutive gps locations."""
    gdf_shifted = gdf.shift()
    consecutive_dist_m = gdf.distance(gdf_shifted, align=False)
    consecutive_time_s = gdf[time_col] - gdf_shifted[time_col]
    consecutive_speed_s = [d/t for (d,t) in zip(consecutive_dist_m, consecutive_time_s)]
    return consecutive_dist_m, consecutive_time_s, consecutive_speed_s


def get_closest_point(points, query_points):
    tree = KDTree(points)
    dists, idxs = tree.query(query_points)
    return dists, idxs


def get_adjacent_metric(shingle_group, adj_traces, d_buffer, t_buffer, b_buffer=None, orthogonal=False):
    """
    Calculate adjacent metric for each shingle from all other shingles in adj_traces.
    """
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