import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
from torch.utils.data import DataLoader

from openbustools import spatial
from openbustools.traveltime import data_loader


class Trajectory():
    def __init__(self, point_attr, traj_attr, coord_ref_center, epsg, resample_len=None, apply_filter=False) -> None:
        """
        Initialize a Trajectory object.

        Args:
            point_attr (dict): Dictionary of point attributes as arrays.
                - lon (np.array): Longitude values.
                - lat (np.array): Latitude values.
            traj_attr (dict): Dictionary of trajectory attributes.
            coord_ref_center (tuple): Tuple of (x,y) coordinates of the system reference point.
            epsg (int): EPSG code of the coordinate reference system.
            resample_len (int, optional): Length to resample the trajectory to. Defaults to None.
            apply_filter (bool, optional): Whether to apply a filter to the trajectory. Defaults to False.
        """
        self.point_attr = point_attr
        self.traj_attr = traj_attr
        self.coord_ref_center = coord_ref_center
        self.epsg = epsg
        self.traj_len = resample_len if resample_len else len(point_attr['lon'])
        if resample_len:
            for key in self.point_attr.keys():
                self.point_attr[key] = spatial.resample_to_len(self.point_attr[key], resample_len)
        if apply_filter:
            for key in self.point_attr.keys():
                window_len = 50
                polyorder = 3
                self.point_attr[key] = scipy.signal.savgol_filter(self.point_attr[key], window_length=window_len, polyorder=polyorder)
        # Create GeoDataFrame and calculate metrics
        gdf = self.point_attr
        gdf.update({'geometry': gpd.points_from_xy(self.point_attr['lon'], self.point_attr['lat'])})
        self.gdf = gpd.GeoDataFrame(gdf, crs="EPSG:4326").to_crs(f"EPSG:{epsg}")
        self.gdf['x'] = self.gdf.geometry.x
        self.gdf['y'] = self.gdf.geometry.y
        self.gdf['x_cent'] = self.gdf['x'] - coord_ref_center[0]
        self.gdf['y_cent'] = self.gdf['y'] - coord_ref_center[1]
        self.gdf['calc_dist_m'], self.gdf['calc_bear_d'] = spatial.calculate_gps_metrics(self.gdf, 'lon', 'lat')
        self.gdf = self.gdf.fillna(0)
        # Set predicted travel time values if they are known
        if 'cumul_time_s' in self.point_attr.keys():
            self.pred_time_s = np.diff(self.point_attr['cumul_time_s'], prepend=0)
        else:
            self.pred_time_s = np.nan(self.traj_len)

    def update_predicted_time(self, model):
        """
        Update the predicted time of the trajectory using a given model.

        Args:
            model: The model used for prediction.
        """
        samples = self.to_torch()
        data_loader.normalize_samples(samples, model.config)
        dataset = data_loader.H5Dataset(samples)
        if model.is_nn:
            loader = DataLoader(
                dataset,
                collate_fn=model.collate_fn,
                batch_size=model.batch_size,
                shuffle=False,
                drop_last=False
            )
            trainer = pl.Trainer(logger=False)
            preds_and_labels = trainer.predict(model=model, dataloaders=loader)
        else:
            preds_and_labels = model.predict(dataset, 'h')
        preds = [x['preds_raw'] for x in preds_and_labels]
        self.gdf['calc_time_s'] = preds[0].flatten()
    
    def to_torch(self):
        """
        Convert the trajectory to a torch format.

        Returns:
            dict: A dictionary containing the trajectory features in torch format.
        """
        gdf = self.gdf.copy()
        # Fill modeling features with -1 if not added to trajectory gdf
        for col in data_loader.NUM_FEAT_COLS + data_loader.MISC_CAT_FEATS:
            if col not in gdf.columns:
                gdf[col] = -1
        feats_n = gdf[data_loader.NUM_FEAT_COLS].to_numpy().astype('int32')
        return {0: {'feats_n': feats_n}}

    def to_momentary_drivecycle(self):
        """
        Convert the trajectory to a DriveCycle object.

        Returns:
            DriveCycle: A DriveCycle object representing the trajectory.
        """
        accel = self.point_attr['measured_accel_m_s2']
        speed = self.point_attr['measured_speed_m_s']
        theta = self.point_attr['measured_theta_d']
        time = self.pred_time_s
        distance = self.gdf['calc_dist_m'].to_numpy()
        cycle = MomentaryDriveCycle(speed, accel, theta, time, distance)
        return cycle


class DriveCycle():
    """Base class for drive cycles. Minimum features to calculate power."""
    def __init__(self, velocity, acceleration, theta):
        self.velocity = velocity
        self.acceleration = acceleration
        self.theta = theta
    def to_df(self):
        pass


class MomentaryDriveCycle(DriveCycle):
    """Acceleration and velocity measured at each point in the trajectory."""
    def __init__(self, velocity, acceleration, theta, time, distance):
        super().__init__(velocity, acceleration, theta)
        self.velocity = velocity
        self.acceleration = acceleration
        self.theta = theta
        self.time = time
        self.distance = distance
    def to_df(self):
        df = pd.DataFrame({
            'Velocity': self.velocity,
            'Acceleration': self.acceleration,
            'Theta': self.theta,
            'Time': self.time,
            'Distance': self.distance
        })
        return df