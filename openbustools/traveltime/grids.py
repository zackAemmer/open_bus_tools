import numpy as np
from collections import defaultdict
import pandas as pd


class NGridBetter:
    def __init__(self, grid_bounds, grid_s_size):
        self.grid_bounds = grid_bounds
        self.grid_s_size = grid_s_size
        self.points = None
        # Create grid boundaries and cells
        x_resolution = (self.grid_bounds[2] - self.grid_bounds[0]) // grid_s_size
        y_resolution = (self.grid_bounds[3] - self.grid_bounds[1]) // grid_s_size
        xbins = np.linspace(self.grid_bounds[0], self.grid_bounds[2], x_resolution)
        # xbins = np.append(xbins, xbins[-1]+.0000001)
        ybins = np.linspace(self.grid_bounds[1], self.grid_bounds[3], y_resolution)
        # ybins = np.append(ybins, ybins[-1]+.0000001)
        self.xbins=xbins
        self.ybins=ybins
        self.cell_lookup = {}
    def digitize_points(self, x_vals, y_vals):
        xbin_idxs = np.digitize(x_vals, self.xbins, right=False)
        ybin_idxs = np.digitize(y_vals, self.ybins, right=False)
        return xbin_idxs, ybin_idxs
    def add_grid_content(self, data, trace_format=False):
        if trace_format:
            data = data[['locationtime','x','y','speed_m_s','bearing']].values
        if self.points is None:
            self.points = data
        else:
            self.points = np.concatenate((self.points, data), axis=0)
    def build_cell_lookup(self):
        # Sort on time from low to high; no longer continuous shingles
        self.points = self.points[np.argsort(self.points[:,0]),:]
        point_xbins, point_ybins = self.digitize_points(self.points[:,1], self.points[:,2])
        # Build lookup table for grid cells to sorted point lists
        lookup = pd.DataFrame(self.points, columns=['locationtime','x','y','speed_m_s','bearing'])
        lookup['xbin'] = point_xbins
        lookup['ybin'] = point_ybins
        self.cell_lookup = lookup.groupby(['xbin','ybin']).apply(lambda x: np.array(x[['locationtime','x','y','speed_m_s','bearing']].values, dtype=float)).to_dict()
    def get_recent_points(self, x_idx, y_idx, locationtime, n_points):
        locationtime_dict = defaultdict(list)
        order_dict = defaultdict(list)
        # Create lookup for unique cells, to time values that will be searched
        for i, (x, y, time) in enumerate(zip(x_idx, y_idx, locationtime)):
            locationtime_dict[(x,y)].append(time)
            order_dict[(x,y)].append(i)
        # Get point values for every unique cell, at the required times
        res_dict = {}
        for k in list(locationtime_dict.keys()):
            # Want to get a set of n_points for every locationtime recorded in this cell
            cell_res = np.full((len(locationtime_dict[k]), n_points, 5), np.nan)
            # Get all points for this grid cell
            cell = self.cell_lookup.get(k, np.array([]))
            if len(cell)!=0:
                # Get the index of each locationtime that we need for this grid cell
                t_idxs = np.searchsorted(cell[:,0], np.array(locationtime_dict[k]))
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
        cell_points = np.full((len(locationtime), n_points, 6), np.nan)
        for k in order_dict.keys():
            loc_order = order_dict[k]
            results = res_dict[k]
            cell_points[loc_order,:,:5] = results
        cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(locationtime),1),n_points,1) - cell_points[:,:,0]
        # TxCxN
        cell_points = np.swapaxes(cell_points, 1, 2)
        return cell_points

# def save_grid_anim(data, file_name):
#     # Plot first 4 channels of second axis
#     fig, axes = plt.subplots(1, data.shape[1])
#     axes = axes.reshape(-1)
#     fig.tight_layout()
#     # Define the update function that will be called for each frame of the animation
#     def update(frame):
#         fig.suptitle(f"Frame {frame}")
#         for i in range(data.shape[1]):
#             d = data[:,i,:,:]
#             vmin=np.min(d[~np.isnan(d)])
#             vmax=np.max(d[~np.isnan(d)])
#             ax = axes[i]
#             ax.clear()
#             im = ax.imshow(data[frame,i,:,:], cmap='plasma', vmin=vmin, vmax=vmax, origin="lower")
#     # Create the animation object
#     ani = animation.FuncAnimation(fig, update, frames=data.shape[0])
#     # Save the animation object
#     ani.save(f"../plots/{file_name}", fps=10, dpi=300)