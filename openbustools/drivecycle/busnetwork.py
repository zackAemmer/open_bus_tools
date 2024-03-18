import geopandas as gpd
import gtfs_kit as gk
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import fastsim as fsim

from openbustools import spatial, trackcleaning, standardfeeds
from openbustools.drivecycle import trajectory
from openbustools.traveltime import data_loader


def get_trajectories(static_dir, epsg, coord_ref_center, dem_file, point_sep_m=300):
    # Load a static feed and break each shape into regularly spaced points
    static_feed = gk.read_feed(static_dir, dist_units="km")
    route_shape_points = standardfeeds.segmentize_route_shapes(static_feed, epsg=epsg, point_sep_m=point_sep_m)
    # Create trajectory for each shape
    trajectories = []
    for shape_id, df in route_shape_points.items():
        traj = trajectory.Trajectory(
            point_attr={
                "lon": df.to_crs(4326).geometry.x.to_numpy(),
                "lat": df.to_crs(4326).geometry.y.to_numpy(),
                "seq_id": df.seq_id.to_numpy(),
            },
            traj_attr={
                'shape_id': shape_id,
                "coord_ref_center": coord_ref_center,
                "epsg": epsg,
                "dem_file": dem_file,
                "t_min_of_day": 9*60,
                "t_day_of_week": 4,
            },
            resample=False
        )
        if len(traj.gdf) > 10:
            trajectories.append(traj)
        else:
            print(f"Skipping shape {shape_id} with only {len(traj.gdf)} points")
    return trajectories


def update_travel_times(trajectories, model):
    # Fill trajectory travel time based on predicted speeds
    preds = trajectory.predict_speeds(trajectories, model)
    res = []
    for batch in preds:
        batch['mask'][0,:] = True
        pred_times = [batch['preds_seq'][:,i][batch['mask'][:,i]] for i in range(batch['preds_seq'].shape[1])]
        res.extend(pred_times)
    dists = [traj.gdf['calc_dist_m'].to_numpy() for traj in trajectories]
    pred_speeds = [dists[i] / res[i] for i in range(len(res))]
    for i, traj in enumerate(trajectories):
        traj.gdf['pred_speed_m_s'] = pred_speeds[i]
        traj.gdf['calc_time_s'] = dists[i] / pred_speeds[i]
        traj.gdf['cumul_time_s'] = traj.gdf['calc_time_s'].cumsum() - traj.gdf['calc_time_s'].iloc[0]


def get_trajectory_energy(traj):
    cycle_pred = {
        "cycGrade": np.clip(spatial.divide_fwd_back_fill(np.diff(traj.gdf['calc_elev_m'], prepend=traj.gdf['calc_elev_m'].iloc[0]), traj.gdf['calc_dist_m']), -0.15, 0.15),
        "mps": spatial.apply_sg_filter(traj.gdf["pred_speed_m_s"].to_numpy(), 8, 0, 30),
        "time_s": traj.gdf['cumul_time_s'].to_numpy(),
        "road_type": np.zeros(len(traj.gdf))
    }
    cycle_pred = fsim.cycle.Cycle.from_dict(fsim.cycle.resample(cycle_pred, new_dt=1)).to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(63, veh_file=Path("..", "data", "FASTSim_py_veh_db.csv")).to_rust()
    sim_drive_pred = fsim.simdrive.RustSimDrive(cycle_pred, veh)
    sim_drive_pred.sim_drive()
    sim_drive_pred = fsim.simdrive.copy_sim_drive(sim_drive_pred, 'python')
    sim_drive_pred = CycleResult(sim_drive_pred)
    return sim_drive_pred


class CycleResult():
    def __init__(self, fastsim_res) -> None:
        # Cycle input variables
        self.cyc = CycleInputResult(fastsim_res)
        # Energy result variables
        self.electric_kwh_per_mi = np.array(fastsim_res.electric_kwh_per_mi)
        self.ess_kw_out_ach = np.array(fastsim_res.ess_kw_out_ach)
        self.accel_kw = np.array(fastsim_res.accel_kw)
        self.rr_kw = np.array(fastsim_res.rr_kw)
        self.ess_loss_kw = np.array(fastsim_res.ess_loss_kw)
        self.ascent_kw = np.array(fastsim_res.ascent_kw)
        self.drag_kw = np.array(fastsim_res.drag_kw)
        self.aux_in_kw = np.array(fastsim_res.aux_in_kw)


class CycleInputResult():
    def __init__(self, fastsim_res) -> None:
        self.time_s = np.array(fastsim_res.cyc.time_s)
        self.mph = np.array(fastsim_res.cyc.mph)
        self.grade = np.array(fastsim_res.cyc.grade)