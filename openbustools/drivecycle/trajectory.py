import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from openbustools import spatial
from openbustools.traveltime import data_loader


class Trajectory():
    def __init__(self, lon, lat, mod, dow, coord_ref_center, epsg, known_times=None, resample_len=None) -> None:
        self.coord_ref_center = coord_ref_center
        self.epsg = epsg
        if resample_len:
            lon_resamp = spatial.resample_to_len(lon, resample_len)
            lat_resamp = spatial.resample_to_len(lat, resample_len)
            self.gdf = gpd.GeoDataFrame({
                'geometry':gpd.points_from_xy(lon_resamp, lat_resamp),
                'lon': lon_resamp,
                'lat': lat_resamp,
                't_min_of_day':mod,
                't_day_of_week':dow}, crs="EPSG:4326").to_crs(f"EPSG:{epsg}")
        else:
            self.gdf = gpd.GeoDataFrame({
                'geometry':gpd.points_from_xy(lon, lat),
                'lon': lon,
                'lat': lat,
                't_min_of_day':mod,
                't_day_of_week':dow}, crs="EPSG:4326").to_crs(f"EPSG:{epsg}")
        self.gdf['x'] = self.gdf.geometry.x
        self.gdf['y'] = self.gdf.geometry.y
        self.gdf['x_cent'] = self.gdf['x'] - coord_ref_center[0]
        self.gdf['y_cent'] = self.gdf['y'] - coord_ref_center[1]
        self.gdf['calc_time_s'] = np.nan
        self.gdf['calc_dist_m'], self.gdf['calc_bear_d'] = spatial.calculate_gps_metrics(self.gdf, 'lon', 'lat')
        self.gdf = self.gdf.drop(index=0)
        self.gdf['calc_time_s'] = 0
        self.gdf['cumul_time_s'] = 0
        self.predicted_time_s = known_times
    def update_predicted_time(self, model):
        samples = self.to_torch()
        data_loader.normalize_samples(samples, model.config, data_loader.LABEL_FEATS+data_loader.GPS_FEATS+data_loader.EMBED_FEATS, data_loader.LABEL_FEATS, data_loader.GPS_FEATS, data_loader.EMBED_FEATS)
        dataset = data_loader.H5Dataset(samples)
        loader = DataLoader(
            dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            shuffle=False,
            drop_last=False
        )
        trainer = pl.Trainer(logger=False)
        preds_and_labels = trainer.predict(model=model, dataloaders=loader)
        preds = [x['preds_raw'] for x in preds_and_labels]
        self.predicted_time_s = preds[0].flatten()
    def to_torch(self):
        feats_n = self.gdf[data_loader.LABEL_FEATS+data_loader.GPS_FEATS+data_loader.EMBED_FEATS].to_numpy().astype('int32')
        return {0: {'feats_n': feats_n}}
    def to_drivecycle(self, dem_path):
        elev = spatial.sample_raster(self.gdf[['x','y']].to_numpy(), dem_path)
        return DriveCycle(self.gdf['calc_dist_m'].to_numpy(), self.predicted_time_s, elev)


class DriveCycle():
    def __init__(self, dist, tim, elev) -> None:
        self.dist = dist
        self.tim = tim
        self.elev = elev
        self.vel = self.dist / self.tim
        # Measured between each point
        self.acc = (np.roll(self.vel, -1) - self.vel) / np.roll(self.tim, -1)
        self.slope = (np.roll(self.elev, -1) - self.elev) / np.clip(np.roll(self.dist, -1), .001, None)
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