import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
from torch.utils.data import DataLoader

from openbustools import spatial
from openbustools.traveltime import data_loader


class Trajectory():
    def __init__(self, point_attr, traj_attr, coord_ref_center, epsg, dem_file=None, resample_len=None, apply_filter=False) -> None:
        """
        Initialize a Trajectory object.

        Args:
            point_attr (dict): Dictionary of point attributes as arrays.
                - lon (np.array): Longitude values.
                - lat (np.array): Latitude values.
            traj_attr (dict): Dictionary of trajectory attributes.
            coord_ref_center (tuple): Tuple of (x,y) coordinates of the system reference point.
            epsg (int): EPSG code of the coordinate reference system.
            dem_file (str): Path to the elevation raster file. Defaults to None.
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
                polyorder = 3
                window_len = max([polyorder + 1, self.traj_len // 20])
                self.point_attr[key] = scipy.signal.savgol_filter(self.point_attr[key], window_length=window_len, polyorder=polyorder)
                if key == 'measured_speed_m_s':
                    self.point_attr[key] = np.clip(self.point_attr[key], a_min=0, a_max=None)
                elif key == 'measured_bear_d':
                    self.point_attr[key][self.point_attr[key]<0] = self.point_attr[key][self.point_attr[key]<0] + 360
                    self.point_attr[key][self.point_attr[key]>360] = self.point_attr[key][self.point_attr[key]>360] - 360
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
        # Get elevation if DEM passed
        if dem_file:
            self.gdf['calc_elev_m'] = spatial.sample_raster(self.gdf[['x','y']].to_numpy(), dem_file)
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
                drop_last=False,
                num_workers=0,
                pin_memory=False
            )
            trainer = pl.Trainer(
                accelerator='cpu',
                logger=False,
                inference_mode=True,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False
            )
            preds_and_labels = trainer.predict(model=model, dataloaders=loader)
        else:
            preds_and_labels = model.predict(dataset, 'h')
        preds = [x['preds_raw'] for x in preds_and_labels]
        self.pred_time_s = preds[0].flatten()

    def to_torch(self):
        """
        Convert the trajectory to format expected for model prediction.

        Returns:
            dict: A dictionary with ordered features in arrays.
        """
        gdf = self.gdf.copy()
        gdf['t_min_of_day'] = self.traj_attr['t_min_of_day']
        gdf['t_day_of_week'] = self.traj_attr['t_day_of_week']
        # Fill modeling features with -1 if not added to trajectory gdf
        for col in data_loader.NUM_FEAT_COLS:
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
        # Time from model or known
        time = self.pred_time_s
        # All motion values taken from point measurements
        accel = self.point_attr['measured_accel_m_s2']
        speed = self.point_attr['measured_speed_m_s']
        theta = self.point_attr['measured_theta_d']
        # Distances from GPS
        distance = self.gdf['calc_dist_m'].to_numpy()
        cycle = MomentaryDriveCycle(speed, accel, theta, time, distance)
        return cycle

    def to_averaged_drivecycle(self):
        """
        Convert the trajectory to a DriveCycle object.

        Returns:
            DriveCycle: A DriveCycle object representing the trajectory.
        """
        # Time from model or known
        time = self.pred_time_s
        # All motion values derived from GPS distances and times
        distance = self.gdf['calc_dist_m'].to_numpy()
        speed = distance[1:] / time[1:]
        accel = np.diff(speed, prepend=0) / time[1:]
        theta = spatial.divide_ffill(np.diff(self.gdf['calc_elev_m'].to_numpy()), distance[1:])
        cycle = MomentaryDriveCycle(speed, accel, theta, time[1:], distance[1:])
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
    """
    Acceleration and velocity measured at each point in the trajectory.
    Time is known or predicted. Distance is from GPS points.
    Velocity, theta and acceleration are used in force calculations.
    Time is used in power calculations.
    
    Attributes:
        velocity (list): List of velocity values.
        acceleration (list): List of acceleration values.
        theta (list): List of theta values.
        time (list): List of time values.
        distance (list): List of distance values.
    """
    def __init__(self, velocity, acceleration, theta, time, distance):
        super().__init__(velocity, acceleration, theta)
        self.velocity = velocity
        self.acceleration = acceleration
        self.theta = theta
        self.time = time
        self.distance = distance

    def to_df(self):
        """Converts the MomentaryDriveCycle object to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing the velocity, acceleration, theta, time, and distance values.
        """
        df = pd.DataFrame({
            'Velocity': self.velocity,
            'Acceleration': self.acceleration,
            'Theta': self.theta,
            'Time': self.time,
            'Distance': self.distance
        })
        return df