from datetime import datetime
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


def get_trajectories(static_dir, target_day, epsg, coord_ref_center, dem_file, point_sep_m=300, min_traj_n=4):
    # Load static feed restricted to target day
    static_feed = gk.read_feed(static_dir, dist_units="km").restrict_to_dates([str.replace(target_day,'_','')])
    # Filter trips to active day of week
    t_day_of_week = datetime.strptime(target_day, "%Y_%m_%d").weekday()
    active_service_ids = static_feed.calendar[static_feed.calendar.iloc[:,t_day_of_week+1]==1]['service_id'].to_numpy()
    trips = static_feed.get_trips()
    trips = trips[trips['service_id'].isin(active_service_ids)]
    # Get starting minute for trips
    stop_times = static_feed.get_stop_times()
    stop_times['t_hour'] = stop_times['departure_time'].str.split(':').str[0].astype(int)
    stop_times.loc[stop_times['t_hour'] > 23, 't_hour'] = stop_times.loc[stop_times['t_hour'] > 23, 't_hour'] - 24
    stop_times['t_min'] = stop_times['departure_time'].str.split(':').str[1].astype(int)
    stop_times['t_min_of_day'] = stop_times['t_hour'] * 60 + stop_times['t_min']
    stop_times = stop_times.sort_values('stop_sequence').groupby('trip_id').first().reset_index()[['trip_id','t_min_of_day']]
    trips = trips.merge(stop_times, on='trip_id')
    trips['t_day_of_week'] = t_day_of_week
    # Break each shape into regularly spaced points
    route_shape_points = standardfeeds.segmentize_shapes(static_feed, epsg=epsg, point_sep_m=point_sep_m)
    # Assemble trajectory for each trip from trip information and shape geometry
    trajectories = []
    for trip_id, df in trips.groupby('trip_id'):
        shape_df = route_shape_points[df['shape_id'].iloc[0]]
        traj = trajectory.Trajectory(
            point_attr={
                'lon': shape_df.to_crs(4326).geometry.x.to_numpy(),
                'lat': shape_df.to_crs(4326).geometry.y.to_numpy(),
                'seq_id': shape_df.seq_id.to_numpy(),
            },
            traj_attr={
                'trip_id': trip_id,
                'shape_id': df['shape_id'].iloc[0],
                'route_id': df['route_id'].iloc[0],
                'direction_id': df['direction_id'].iloc[0],
                'block_id': df['block_id'].iloc[0],
                'service_id': df['service_id'].iloc[0],
                'coord_ref_center': coord_ref_center,
                'epsg': epsg,
                'dem_file': dem_file,
                't_min_of_day': df['t_min_of_day'].iloc[0],
                't_day_of_week': df['t_day_of_week'].iloc[0],
            },
            resample=False
        )
        if len(traj.gdf) >= min_traj_n:
            trajectories.append(traj)
        else:
            print(f"Skipping trip {trip_id} with only {len(traj.gdf)} points")
    return trajectories


def predict_speeds(trajectories, model):
    dataset = data_loader.trajectoryDataset(trajectories, model.config)
    if model.is_nn:
        if torch.cuda.is_available():
            num_workers = 4
            pin_memory = True
            accelerator = "cuda"
        else:
            num_workers = 0
            pin_memory = False
            accelerator = "cpu"
        loader = DataLoader(
            dataset,
            collate_fn=model.collate_fn,
            batch_size=model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            logger=False,
            inference_mode=True,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        preds_and_labels = trainer.predict(model=model, dataloaders=loader)
    else:
        preds_and_labels = model.predict(dataset)
    return preds_and_labels


def update_travel_times(trajectories, model):
    # Replace trajectory travel time based on predicted speeds
    preds = predict_speeds(trajectories, model)
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


def get_trajectory_energy(traj, veh_file):
    cycle_pred = {
        "cycGrade": np.clip(spatial.divide_fwd_back_fill(np.diff(traj.gdf['calc_elev_m'], prepend=traj.gdf['calc_elev_m'].iloc[0]), traj.gdf['calc_dist_m']), -0.15, 0.15),
        "mps": spatial.apply_sg_filter(traj.gdf["pred_speed_m_s"].to_numpy(), polyorder=8, clip_min=0, clip_max=30),
        "time_s": traj.gdf['cumul_time_s'].to_numpy(),
        "road_type": np.zeros(len(traj.gdf))
    }
    cycle_pred = fsim.cycle.Cycle.from_dict(fsim.cycle.resample(cycle_pred, new_dt=1)).to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(63, veh_file=veh_file).to_rust()
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