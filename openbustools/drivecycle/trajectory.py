import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
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
            # Assumes there are timestamps in point_attr
            df.index = pd.to_datetime(df['locationtime'], unit='s')
            df.index.name = 'time'
            df = df.drop(columns=['locationtime'])
            df = df.resample('s').mean().interpolate('linear')
            df['locationtime'] = df.index.astype(int) // 10**9
        if 'locationtime' in self.point_attr.keys():
            df['calc_dist_m'],  df['calc_bear_d'],  df['calc_time_s'] = spatial.calculate_gps_metrics(df, 'lon', 'lat', time_col='locationtime')
            df['calc_speed_m_s'] = df['calc_dist_m'] / df['calc_time_s']
        else:
            df['calc_dist_m'], df['calc_bear_d'] = spatial.calculate_gps_metrics(df, 'lon', 'lat')
            df['calc_time_s'] = np.zeros(len(df))
        df['cumul_time_s'] = df['locationtime'] - self.traj_attr['start_epoch']
        df['shingle_id'] = 0
        df = df.ffill().bfill()
        # Add calculated values from geometry, timestamps
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs=4326).to_crs(self.traj_attr['epsg'])
        self.gdf['x'] = self.gdf.geometry.x
        self.gdf['y'] = self.gdf.geometry.y
        self.gdf['x_cent'] = self.gdf['x'] - self.traj_attr['coord_ref_center'][0]
        self.gdf['y_cent'] = self.gdf['y'] - self.traj_attr['coord_ref_center'][1]
        self.gdf['calc_elev_m'] = spatial.sample_raster(self.gdf[['x','y']].to_numpy(), self.traj_attr['dem_file'])
    # def to_fastsim_cycle(self):
    #     """
    #     Convert the trajectory to a FastSim cycle; use measured values if available.
    #     """
    #     # Distance in m
    #     smoothed = spatial.apply_sg_filter(gdf['calc_speed_m_s'].to_numpy(), 8, 0, 30)
    #     distance = self.gdf['calc_dist_m'].to_numpy()
    #     # Time in seconds
    #     time = self.gdf['calc_time_s'].to_numpy()
    #     # Speed in m/s
    #     if "measured_speed_m_s" in self.point_attr.keys():
    #         speed = self.point_attr['measured_speed_m_s']
    #     else:
    #         # All motion values derived from GPS distances and times
    #         speed = spatial.divide_fwd_back_fill(distance, time)
    #     # Grade in %/100
    #     if "measured_elev_m" in self.point_attr.keys():
    #         elev = self.point_attr['measured_elev_m']
    #     else:
    #         elev = self.gdf['elev_m'].to_numpy()
    #     grade = spatial.divide_fwd_back_fill(np.diff(elev, prepend=elev[0]), distance)
    #     grade = np.clip(grade, a_min=-0.25, a_max=0.25)
    #     # Dictionary of cycle values for Fastsim
    #     cycle = {
    #         'cycGrade': grade,
    #         'mps': speed,
    #         'time_s': np.cumsum(time),
    #         'road_type': np.zeros(self.traj_len),
    #     }
    #     return cycle


def predict_trajectory_times(trajectories, model):
    """
    Update the predicted time of the trajectory using a given model.

    Args:
        model: The model used for prediction.
    """
    dataset = data_loader.trajectoryDataset(trajectories, model.config)
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
        preds_and_labels = model.predict(dataset)
    preds = [x['preds_seq'].flatten() for x in preds_and_labels]
    return preds