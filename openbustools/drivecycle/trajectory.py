import geopandas as gpd
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
from torch.utils.data import DataLoader

from openbustools import spatial, trackcleaning
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
        # if apply_filter:
        #     for key in self.point_attr.keys():
        #         polyorder = 3
        #         window_len = max([polyorder + 1, self.traj_len // 20])
        #         self.point_attr[key] = scipy.signal.savgol_filter(self.point_attr[key], window_length=window_len, polyorder=polyorder)
        #         if key == 'measured_speed_m_s':
        #             self.point_attr[key] = np.clip(self.point_attr[key], a_min=0, a_max=None)
        #         elif key == 'measured_bear_d':
        #             self.point_attr[key][self.point_attr[key]<0] = self.point_attr[key][self.point_attr[key]<0] + 360
        #             self.point_attr[key][self.point_attr[key]>360] = self.point_attr[key][self.point_attr[key]>360] - 360
        # Create GeoDataFrame and calculate metrics
        gdf = self.point_attr.copy()
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
            self.gdf['elev_m'] = spatial.sample_raster(self.gdf[['x','y']].to_numpy(), dem_file)
        # Set predicted travel time values if they are known
        if 'cumul_time_s' in self.point_attr.keys():
            self.pred_time_s = np.diff(self.point_attr['cumul_time_s'], prepend=0)
        else:
            self.pred_time_s = np.repeat(np.nan, self.traj_len)
    def to_fastsim_cycle(self):
        """
        Convert the trajectory to a FastSim cycle; use measured values if available.
        """
        # Distance in m
        distance = self.gdf['calc_dist_m'].to_numpy()
        # Time in seconds
        time = self.pred_time_s
        # Speed in m/s
        if "measured_speed_m_s" in self.point_attr.keys():
            speed = self.point_attr['measured_speed_m_s']
        else:
            # All motion values derived from GPS distances and times
            speed = spatial.divide_fwd_back_fill(distance, time)
        # Grade in %/100
        if "measured_elev_m" in self.point_attr.keys():
            elev = self.point_attr['measured_elev_m']
        else:
            elev = self.gdf['elev_m'].to_numpy()
        grade = spatial.divide_fwd_back_fill(np.diff(elev, prepend=elev[0]), distance)
        grade = np.clip(grade, a_min=-0.25, a_max=0.25)
        # Dictionary of cycle values for Fastsim
        cycle = {
            'cycGrade': grade,
            'mps': speed,
            'time_s': np.cumsum(time),
            'road_type': np.zeros(self.traj_len),
        }
        return cycle


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