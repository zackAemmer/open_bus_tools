import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from openbustools import spatial
from openbustools.traveltime import data_loader


class Trajectory():
    def __init__(self, lon, lat, mod, dow, coord_ref_center, epsg, known_times=None, resample_len=None) -> None:
        """
        Initialize a Trajectory object.

        Args:
            lon (list): List of longitude values.
            lat (list): List of latitude values.
            mod (int): Time of the trajectory in minutes.
            dow (int): Day of the week (0-6, Monday to Sunday).
            coord_ref_center (tuple): Tuple of the reference center coordinates (x, y).
            epsg (int): EPSG code for the coordinate reference system.
            known_times (list, optional): List of cumulative time in seconds. Defaults to None.
            resample_len (int, optional): Length to resample the trajectory to. Defaults to None.
        """
        self.coord_ref_center = coord_ref_center
        self.epsg = epsg
        self.predicted_time_s = None
        if resample_len is not None:
            lon = spatial.resample_to_len(lon, resample_len)
            lat = spatial.resample_to_len(lat, resample_len)
        self.gdf = gpd.GeoDataFrame({
            'geometry': gpd.points_from_xy(lon, lat),
            'lon': lon,
            'lat': lat,
            't_hour': mod // 60,
            't_min_of_day':mod,
            't_day_of_week':dow}, crs="EPSG:4326").to_crs(f"EPSG:{epsg}")
        self.gdf['x'] = self.gdf.geometry.x
        self.gdf['y'] = self.gdf.geometry.y
        self.gdf['x_cent'] = self.gdf['x'] - coord_ref_center[0]
        self.gdf['y_cent'] = self.gdf['y'] - coord_ref_center[1]
        if known_times is not None:
            if resample_len is not None:
                known_times = spatial.resample_to_len(known_times, resample_len)
            self.gdf['cumul_time_s'] = known_times
            self.gdf['calc_time_s'] = self.gdf['cumul_time_s'].diff()
            self.gdf['calc_time_s'] = self.gdf['calc_time_s'].fillna(1)
        self.gdf['calc_dist_m'], self.gdf['calc_bear_d'] = spatial.calculate_gps_metrics(self.gdf, 'lon', 'lat')
        self.gdf = self.gdf.drop(index=0)
    
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
        for col in data_loader.NUM_FEAT_COLS+data_loader.MISC_CAT_FEATS:
            if col not in gdf.columns:
                gdf[col] = -1
        feats_n = gdf[data_loader.NUM_FEAT_COLS].to_numpy().astype('int32')
        return {0: {'feats_n': feats_n}}
    
    def to_average_drivecycle(self, dem_path):
        """
        Convert the trajectory to a DriveCycle object.

        Args:
            dem_path (str): Path to the digital elevation model (DEM) file.

        Returns:
            DriveCycle: A DriveCycle object representing the trajectory.
        """
        elev = spatial.sample_raster(self.gdf[['x','y']].to_numpy(), dem_path)
        return AverageDriveCycle(self.gdf['calc_dist_m'].to_numpy(), self.gdf['calc_time_s'].to_numpy(), elev)

    def to_momentary_drivecycle(self, vel, acc, theta):
        """
        Convert the trajectory to a DriveCycle object.

        Returns:
            DriveCycle: A DriveCycle object representing the trajectory.
        """
        return MomentaryDriveCycle(vel, acc, theta, self.gdf['calc_dist_m'].to_numpy(), self.gdf['calc_time_s'].to_numpy())


class AverageDriveCycle():
    def __init__(self, dist, tim, elev) -> None:
        self.dist = dist
        self.tim = tim
        self.elev = elev
        self.vel = self.dist / self.tim
        # Measured between each point
        self.acc = np.diff(self.vel) / self.tim[1:]
        self.slope = np.diff(self.elev) / np.clip(self.dist[1:], .001, None)
        self.theta = np.arctan(self.slope)
        # Calculated new avg values; lose first point
        self.dist = self.dist[1:]
        self.tim = self.tim[1:]
        self.elev = self.elev[1:]
        self.vel = self.vel[1:]
        self.traj_len = len(self.dist)
    def to_df(self):
        df = pd.DataFrame({
            'Distance': self.dist,
            'Time': self.tim,
            'Elevation': self.elev,
            'Velocity': self.vel,
            'Acceleration': self.acc,
            'Slope': self.slope,
            'Theta': self.theta})
        return df


class MomentaryDriveCycle():
    def __init__(self, vel, acc, theta, dist, tim) -> None:
        self.vel = vel
        self.acc = acc
        self.dist = dist
        self.tim = tim
        self.theta = theta
        self.traj_len = len(self.vel)
    def to_df(self):
        df = pd.DataFrame({
            'Distance': self.dist,
            'Time': self.tim,
            'Velocity': self.vel,
            'Acceleration': self.acc,
            'Theta': self.theta})
        return df


class RandomDriveCycle():
    def __init__(self) -> None:
        self.traj_len = np.random.randint(10,50)
        self.vel = np.random.randint(1, 35, size=self.traj_len)
        self.dist = np.random.randint(1, 4200, size=self.traj_len)
        self.tim = self.dist / self.vel
        self.elev = np.random.randint(-200, 200, size=self.traj_len)
        # Measured between each point
        self.acc =( np.roll(self.vel, -1) - self.vel) / np.roll(self.tim, -1)
        self.slope = np.roll(self.elev, -1) - self.elev / np.roll(self.dist, -1)
        self.theta = np.arctan(self.slope)
        # Calculated new avg values; lose first point
        self.dist = self.dist[:-1]
        self.tim = self.tim[:-1]
        self.elev = self.elev[:-1]
        self.vel = self.vel[:-1]
        self.acc = self.acc[:-1]
        self.slope = self.slope[:-1]
        self.theta = self.theta[:-1]
        self.traj_len = len(self.dist)
    def to_df(self):
        df = pd.DataFrame({
            'Distance': self.dist,
            'Time': self.tim,
            'Elevation': self.elev,
            'Velocity': self.vel,
            'Acceleration': self.acc,
            'Slope': self.slope,
            'Theta': self.theta})
        return df