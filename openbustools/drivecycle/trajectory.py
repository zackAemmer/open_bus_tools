import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
import torch
from torch.utils.data import DataLoader

from openbustools import spatial, trackcleaning
from openbustools.traveltime import data_loader


class Trajectory():
    def __init__(self, point_attr, traj_attr, resample=False) -> None:
        """
        Initialize a Trajectory object. Save initial values then create cleaned GeoDataFrame.

        Args:
            point_attr (dict): Dictionary of point attributes as arrays.
                - lon (np.array): Longitude values.
                - lat (np.array): Latitude values.
            traj_attr (dict): Dictionary of trajectory attributes.
                - coord_ref_center (tuple): Tuple of (x,y) coordinates of the system reference point.
                - epsg (int): EPSG code of the coordinate reference system.
                - dem_file (str): Path to the digital elevation model file.
        """
        # Store passed trajectory attributes
        self.point_attr = point_attr
        self.traj_attr = traj_attr
        # Create GeoDataFrame with modified values
        df = pd.DataFrame(self.point_attr.copy())
        if resample:
            # Used in matching realtime to phone/gnss
            # Assumes there are timestamps in point_attr
            df.index = pd.to_datetime(df['locationtime'], unit='s')
            df.index.name = 'time'
            df = df.drop(columns=['locationtime'])
            df = df.resample('s').mean().interpolate('linear')
            df['locationtime'] = df.index.astype(int) // 10**9
        if 'locationtime' in self.point_attr.keys():
            df['calc_dist_m'],  df['calc_bear_d'],  df['calc_time_s'] = spatial.calculate_gps_metrics(df, 'lon', 'lat', time_col='locationtime')
            df['calc_speed_m_s'] = df['calc_dist_m'] / df['calc_time_s']
            df['cumul_time_s'] = df['locationtime'] - self.traj_attr['start_epoch']
        else:
            df['calc_dist_m'], df['calc_bear_d'] = spatial.calculate_gps_metrics(df, 'lon', 'lat')
            df['calc_time_s'] = np.zeros(len(df))
            df['calc_speed_m_s'] = np.zeros(len(df))
            df['cumul_time_s'] = np.ones(len(df))
        df['shingle_id'] = 0
        df = df.ffill().bfill()
        # Add calculated values from geometry, timestamps
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs=4326).to_crs(self.traj_attr['epsg'])
        self.gdf['x'] = self.gdf.geometry.x
        self.gdf['y'] = self.gdf.geometry.y
        self.gdf['x_cent'] = self.gdf['x'] - self.traj_attr['coord_ref_center'][0]
        self.gdf['y_cent'] = self.gdf['y'] - self.traj_attr['coord_ref_center'][1]
        self.gdf['calc_elev_m'] = spatial.sample_raster(self.gdf[['x','y']].to_numpy(), self.traj_attr['dem_file'])