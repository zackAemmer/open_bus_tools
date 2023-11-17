import datetime

import numpy as np
from collections import defaultdict
import pandas as pd


class RealtimeGrid:
    def __init__(self, grid_bounds, grid_s_size):
        self.grid_bounds = grid_bounds
        self.grid_s_size = grid_s_size
        self.points = None
        # Create grid boundaries and cells
        x_resolution = (self.grid_bounds[2] - self.grid_bounds[0]) // grid_s_size
        y_resolution = (self.grid_bounds[3] - self.grid_bounds[1]) // grid_s_size
        xbins = np.linspace(self.grid_bounds[0], self.grid_bounds[2], x_resolution)
        ybins = np.linspace(self.grid_bounds[1], self.grid_bounds[3], y_resolution)
        self.xbins=xbins
        self.ybins=ybins
        self.cell_lookup = {}
        self.column_names = []
    def digitize_points(self, x_vals, y_vals):
        xbin_idxs = np.digitize(x_vals, self.xbins, right=False)
        ybin_idxs = np.digitize(y_vals, self.ybins, right=False)
        return xbin_idxs, ybin_idxs
    def build_cell_lookup(self, points_df):
        # Sort on time from low to high; no longer continuous shingles
        points_df = points_df.sort_values('locationtime')
        points_df['xbin'], points_df['ybin'] = self.digitize_points(points_df['x'].to_numpy(), points_df['y'].to_numpy())
        self.cell_lookup = points_df.groupby(['xbin','ybin']).apply(lambda x: x.to_numpy()).to_dict()
        self.column_names.extend(list(points_df.columns))
        self.column_names.extend(['elapsed_s'])
    def get_recent_points(self, sample_df, n_points):
        locationtime_dict = defaultdict(list)
        order_dict = defaultdict(list)
        dim_feats = self.cell_lookup[list(self.cell_lookup.keys())[0]].shape[1]
        x_idx, y_idx = self.digitize_points(sample_df[:,0], sample_df[:,1])
        # Only get points before the timestamp of the first point in trajectory
        locationtimes = np.repeat(sample_df[0,2], sample_df.shape[0])
        # Create lookup for unique cells, to time values that will be searched
        for i, (x, y, time) in enumerate(zip(x_idx, y_idx, locationtimes)):
            locationtime_dict[(x,y)].append(time)
            order_dict[(x,y)].append(i)
        # Get point values for every unique cell, at the required times
        res_dict = {}
        for k in list(locationtime_dict.keys()):
            # Want to get a set of n_points for every locationtime recorded in this cell
            cell_res = np.full((len(locationtime_dict[k]), n_points, dim_feats), np.nan)
            # Get all points for this grid cell
            cell = self.cell_lookup.get(k, np.array([]))
            if len(cell)!=0:
                # Get the index of each locationtime that we need for this grid cell
                t_idxs = np.searchsorted(cell[:,0], np.array(locationtime_dict[k])) - 1
                # Record the index through index-n_points for each locationtime that we need for this grid cell
                for i,n_back in enumerate(range(n_points)):
                    idx_back = t_idxs - n_back
                    # Record which points should be filled with nan
                    mask = idx_back < 0
                    # Clip so that operation can still be performed
                    idx_back = np.clip(idx_back, a_min=0, a_max=len(cell)-1)
                    cell_res[:,i,:] = cell[idx_back]
                    # Fill nans (instead of repeating the first cell value), this is more informative
                    cell_res[mask] = np.nan
                # Save all cell results
            res_dict[k] = cell_res
        # Reconstruct final result in the correct order (original locationtimes have been split among dict keys)
        cell_points = np.full((sample_df.shape[0], n_points, dim_feats+1), np.nan)
        for k in order_dict.keys():
            loc_order = order_dict[k]
            results = res_dict[k]
            cell_points[loc_order,:,:-1] = results
        # Time difference between starting point and grid observations
        # This requires locationtime to be the first column
        cell_points[:,:,-1] = cell_points[:,:,0] - np.repeat(np.expand_dims(np.array(locationtimes),1),n_points,1)
        # Fill nans w/feature averages
        if np.isnan(cell_points).all():
            cell_points = np.nan_to_num(cell_points, nan=0.0)
        elif np.isnan(cell_points).any():
            feat_means = np.nanmean(cell_points, axis=(0,1))
            for i, val in enumerate(feat_means):
                cell_points[:,:,i] = np.nan_to_num(cell_points[:,:,i], nan=val)
        # Sample x Channels x Points
        cell_points = np.swapaxes(cell_points, 1, 2)
        return cell_points


def convert_to_frames(g):
    # Cell lookup to dataframe of points
    all = []
    for k,val in g.cell_lookup.items():
        all.append(val)
    all = np.concatenate(all, axis=0)
    all = pd.DataFrame(all)
    all.columns = g.column_names[:-1]
    # Dataframe of points to grouped tbin/xbin/ybins
    t_bins = np.arange(0,1*60*24)
    timestamps = [datetime.datetime.fromtimestamp(x) for x in all['locationtime'].to_numpy().astype(int)]
    timestamps = [t.hour*60+t.minute for t in timestamps]
    t_idxs = np.digitize(timestamps, t_bins)
    all['tbin'] = t_idxs
    all = all.sort_values('locationtime')
    all = all.groupby(['tbin','ybin','xbin']).mean()[['calc_speed_m_s']].reset_index()
    all = all.to_numpy()
    # Grouped bins to sparse 3d array
    res = np.full((len(t_bins)+1,g.ybins.shape[0]+1,g.xbins.shape[0]+1), np.nan)
    for i in range(all.shape[0]):
        row = all[i,:]
        res[row[0].astype(int),row[1].astype(int),row[2].astype(int)] = row[3]
    return res