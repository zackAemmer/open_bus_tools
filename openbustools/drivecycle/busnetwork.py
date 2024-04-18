from datetime import datetime
import geopandas as gpd
import gtfs_kit as gk
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import fastsim as fsim

from openbustools import spatial, trackcleaning, standardfeeds
from openbustools.drivecycle import trajectory
from openbustools.traveltime import data_loader


CABIN_TEMP_F = 65
AIR_DENSITY_LB_FT3 = 0.075
AIR_SPECIFIC_HEAT_BTU_LBDEG = 0.24
HEAT_EFF = 0.95
COOL_EFF = 0.90
CABIN_VOLUME_FT3 = 10*40*8
# Multiply by temperature differential in F to get kWh:
TOTAL_CABIN_WARMUP_KWH = CABIN_VOLUME_FT3*AIR_DENSITY_LB_FT3*AIR_SPECIFIC_HEAT_BTU_LBDEG/HEAT_EFF/3412

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
    stop_times_start = stop_times.sort_values('stop_sequence').groupby('trip_id').first().reset_index()[['trip_id','t_min_of_day']]
    stop_times_end = stop_times.sort_values('stop_sequence').groupby('trip_id').last().reset_index()[['trip_id','t_min_of_day']]
    stop_times_count = stop_times.groupby('trip_id').size().reset_index(name='stop_count')
    trips = trips.merge(stop_times_start, on='trip_id', suffixes=('','_start'))
    trips = trips.merge(stop_times_end, on='trip_id', suffixes=('','_end'))
    trips = trips.merge(stop_times_count, on='trip_id')
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
                't_min_of_day_end': df['t_min_of_day_end'].iloc[0],
                't_day_of_week': df['t_day_of_week'].iloc[0],
                'stop_count': df['stop_count'].iloc[0],
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


def get_trajectory_energy(traj, veh_file, veh_num, passenger_load, aux_power_kw, acc_dec_factor):
    # Incorporate acceleration/deceleration parameter
    filtered_mps = spatial.apply_peak_filter(traj.gdf['pred_speed_m_s'].to_numpy(), window_len=3, scalar=acc_dec_factor, clip_min=0, clip_max=35)
    filtered_mps = spatial.apply_sg_filter(filtered_mps, clip_min=0, clip_max=35)
    cycle_pred = {
        "cycGrade": np.clip(spatial.divide_fwd_back_fill(np.diff(traj.gdf['calc_elev_m'], prepend=traj.gdf['calc_elev_m'].iloc[0]), traj.gdf['calc_dist_m']), -0.15, 0.15),
        "mps": filtered_mps,
        "time_s": traj.gdf['cumul_time_s'].to_numpy(),
        "road_type": np.zeros(len(traj.gdf))
    }
    cycle_pred = fsim.cycle.Cycle.from_dict(fsim.cycle.resample(cycle_pred, new_dt=1)).to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(veh_num, veh_file=veh_file).to_rust()
    # Incorporate static aux power parameter
    veh.aux_kw = aux_power_kw
    # Incorporate passenger load parameter
    veh.veh_kg = veh.veh_kg + (150 * passenger_load * 0.453592)
    sim_drive_pred = fsim.simdrive.RustSimDrive(cycle_pred, veh)
    sim_drive_pred.sim_drive()
    sim_drive_pred = fsim.simdrive.copy_sim_drive(sim_drive_pred, 'python')
    sim_drive_pred = CycleResult(sim_drive_pred)
    return sim_drive_pred


def get_energy_results(network_trajs, network_cycles, n_depots, deadhead_economy_kwh_mi, deadhead_aux_kw, temperature_f, door_open_time_s, diesel_heater, preconditioning):
    """
    Post processing for energy results.
    """
    # Pull info from the trajectories (input) and cycles (fastsim outputs)
    network_id = network_trajs[0].traj_attr['dem_file'].parent.name.split("_")[0]
    trip_ids = [t.traj_attr['trip_id'] for t in network_trajs]
    block_ids = [t.traj_attr['block_id'] for t in network_trajs]

    # If the agency doesn't report blocks treat each trip as a block
    if pd.isna(block_ids).any():
        block_ids = np.arange(len(trip_ids))
    num_trips = len(trip_ids)
    num_blocks = len(set(block_ids))
    trip_economy = [float(cycle.electric_kwh_per_mi) for cycle in network_cycles]
    trip_distance = [int(np.sum(t.gdf['calc_dist_m'])) for t in network_trajs]
    start_loc = [t.gdf.iloc[0].geometry for t in network_trajs]
    end_loc = [t.gdf.iloc[-1].geometry for t in network_trajs]
    t_min_of_day = [t.traj_attr['t_min_of_day'] for t in network_trajs]
    t_min_of_day_end = [t.traj_attr['t_min_of_day_end'] for t in network_trajs]

    # Energy use and regeneration for drive cycle and estimate of net energy for block
    total_cyc_regen_kwh = [np.sum(cycle.ess_kw_out_ach[cycle.ess_kw_out_ach<0]) / 3600 for cycle in network_cycles]
    total_cyc_driven_kwh = [np.sum(cycle.ess_kw_out_ach[cycle.ess_kw_out_ach>0]) / 3600 for cycle in network_cycles]
    total_cyc_kwh = [np.sum(cycle.ess_kw_out_ach) / 3600 for cycle in network_cycles]
    trip_estimated_kwh = [cycle.electric_kwh_per_mi * (trip_distance[i] / 1000 / 1.609) for i, cycle in enumerate(network_cycles)]

    # Create a dataframe with rows for each trip in the network on the target day
    df = pd.DataFrame({
        'network_id': network_id,
        'block_id': block_ids,
        'trip_id': trip_ids,
        't_min_of_day': t_min_of_day,
        't_min_of_day_end': t_min_of_day_end,
        'economy_kwh_mi': trip_economy,
        'trip_distance_m': trip_distance,
        'start_loc': start_loc,
        'end_loc': end_loc,
        'num_trips': num_trips,
        'num_blocks': num_blocks,
        'cyc_regen_kwh': total_cyc_regen_kwh,
        'cyc_driven_kwh': total_cyc_driven_kwh,
        'cyc_kwh': total_cyc_kwh,
        'trip_estimated_kwh': trip_estimated_kwh
    })
    # Allow times to cross midnight; were changed for travel time prediction
    df.loc[df['t_min_of_day_end'] <= df['t_min_of_day'], 't_min_of_day_end'] += 1440
    df = df.sort_values(['network_id','block_id','t_min_of_day'])
    # Get time of trips and down time between trips
    df['t_min_trip'] = df['t_min_of_day_end'] - df['t_min_of_day']
    df['t_min_down'] = df.groupby('block_id').apply(func=lambda x: x['t_min_of_day'].shift(-1, fill_value=x['t_min_of_day_end'].iloc[-1]) - x['t_min_of_day_end'], include_groups=False).to_numpy()
    # Get manhattan distance between trips
    df['start_loc_next'] = df.groupby('block_id').apply(func=lambda x: x['start_loc'].shift(-1, fill_value=x['end_loc'].iloc[-1])).to_numpy()
    df['trip_deadhead_dist_m'] = spatial.manhattan_distance(np.array([loc.x for loc in df['start_loc_next']]), np.array([loc.y for loc in df['start_loc_next']]), np.array([loc.x for loc in df['end_loc']]), np.array([loc.y for loc in df['end_loc']]))
    df['trip_deadhead_drive_kwh'] = df['trip_deadhead_dist_m'] / 1000 / 1.609 * deadhead_economy_kwh_mi
    df['trip_deadhead_aux_kwh'] = np.clip(df['t_min_down'], a_min=0, a_max=120) * 60 * deadhead_aux_kw / 3600
    df['trip_deadhead_kwh'] = df['trip_deadhead_drive_kwh'] + df['trip_deadhead_aux_kwh']

    # Calculate energy use for HVAC related to door opening
    temp_differential_f = CABIN_TEMP_F - temperature_f
    if diesel_heater or temp_differential_f == 0:
        tot_door_kwh = np.array([0.0 for t in network_trajs])
    else:
        # Constants
        door_area_ft2 = 8*4*.5
        wind_speed_ft_s = 1.5
        air_density_lb_ft3 = 0.075
        air_specific_heat_btu_lbdeg = 0.24
        heat_eff = 0.95
        cool_eff = 0.90
        # Energy from air loss
        Q_air_ft3_s = door_area_ft2 * wind_speed_ft_s
        Q_air_lb_s = Q_air_ft3_s * air_density_lb_ft3
        Q_air_btu_s = Q_air_lb_s * temp_differential_f * air_specific_heat_btu_lbdeg
        if temp_differential_f > 0:
            hvac_btu_s = Q_air_btu_s / heat_eff
        else:
            hvac_btu_s = Q_air_btu_s / cool_eff
        tot_door_times_s = np.array([t.traj_attr['stop_count'] * door_open_time_s for t in network_trajs])
        tot_door_btu = tot_door_times_s * hvac_btu_s
        tot_door_kwh = tot_door_btu / 3412
    df['trip_door_kwh'] = tot_door_kwh
    # Calculate energy use for HVAC related to initial block warmup
    if not preconditioning and temp_differential_f > 0:
        tot_cabin_warmup_kwh = TOTAL_CABIN_WARMUP_KWH * temp_differential_f
    else:
        tot_cabin_warmup_kwh = 0

    # Get theoretical depot locations with kmeans on block starting locations
    kmeans = KMeans(n_clusters=n_depots, random_state=0, n_init="auto").fit(np.array([(loc.x, loc.y) for loc in df.groupby('block_id').first()['start_loc']]))
    depot_assigned = kmeans.labels_
    depot_locations = kmeans.cluster_centers_
    # Get distance from start and end of each block to its assigned depot
    depot_assignments = pd.DataFrame({
        "block_id": df.groupby('block_id').first().index,
        "block_start_x": [bs.x for bs in df.groupby('block_id').first()['start_loc']],
        "block_start_y": [bs.y for bs in df.groupby('block_id').first()['start_loc']],
        "block_end_x": [bs.x for bs in df.groupby('block_id').last()['end_loc']],
        "block_end_y": [bs.y for bs in df.groupby('block_id').last()['end_loc']],
        "depot_id": depot_assigned,
    })
    depot_locations = pd.DataFrame({
        "depot_id": range(n_depots),
        "depot_x": depot_locations[:,0],
        "depot_y": depot_locations[:,1],
    })
    depot_assignments = pd.merge(depot_assignments, depot_locations, on="depot_id")
    depot_assignments['block_deadhead_start_dist_m'] = depot_assignments.apply(lambda x: spatial.manhattan_distance(x['block_start_x'], x['block_start_y'], x['depot_x'], x['depot_y']), axis=1)
    depot_assignments['block_deadhead_end_dist_m'] = depot_assignments.apply(lambda x: spatial.manhattan_distance(x['block_end_x'], x['block_end_y'], x['depot_x'], x['depot_y']), axis=1)
    depot_assignments['block_deadhead_start_kwh'] = depot_assignments['block_deadhead_start_dist_m'] / 1000 * deadhead_economy_kwh_mi / 1.609
    depot_assignments['block_deadhead_end_kwh'] = depot_assignments['block_deadhead_end_dist_m'] / 1000 * deadhead_economy_kwh_mi / 1.609
    depot_assignments['block_warmup_kwh'] = tot_cabin_warmup_kwh
    depot_assignments = depot_assignments[['block_id','depot_id','block_deadhead_start_kwh','block_deadhead_end_kwh','block_warmup_kwh']].copy()
    df = pd.merge(df, depot_assignments, on="block_id")

    block_total_kwh = df.groupby('block_id', as_index=False).agg({
        'trip_estimated_kwh': 'sum', # Uses actual trip distance and estimated avg economy
        'trip_deadhead_drive_kwh': 'sum', # Distance between trips
        'trip_deadhead_aux_kwh': 'sum', # Time between trips
        'trip_door_kwh': 'sum', # Additional aux from door open/close during trips
        'block_deadhead_start_kwh': 'first', # Both aux and drive
        'block_deadhead_end_kwh': 'first', # Both aux and drive
        'block_warmup_kwh': 'first', # Aux for initial block warmup
        'cyc_regen_kwh': 'sum', # Does not include deadhead drive
        'cyc_driven_kwh': 'sum', # Does not include deadhead drive
        'cyc_kwh': 'sum', # Does not include deadhead drive
    })
    # Assign metrics aligned to ChargePoint Report
    block_total_kwh['Energy charged'] = -1
    block_total_kwh['Energy consumed driving'] = block_total_kwh['cyc_driven_kwh'] + block_total_kwh['trip_deadhead_drive_kwh'] + block_total_kwh['block_deadhead_start_kwh'] + block_total_kwh['block_deadhead_end_kwh'] + block_total_kwh['trip_door_kwh'] + block_total_kwh['block_warmup_kwh']
    block_total_kwh['Energy regenerated driving'] = block_total_kwh['cyc_regen_kwh'] * -1
    block_total_kwh['Energy driven'] = block_total_kwh['Energy consumed driving'] - block_total_kwh['Energy regenerated driving']
    block_total_kwh['Energy idled in service'] = 0
    block_total_kwh['Energy idled not in service'] = block_total_kwh['trip_deadhead_aux_kwh']
    block_total_kwh['Energy used in service'] = block_total_kwh['Energy driven'] + block_total_kwh['Energy idled in service']
    block_total_kwh['Energy used not in service'] = block_total_kwh['Energy idled not in service']
    block_total_kwh['Energy used'] = block_total_kwh['Energy used in service'] + block_total_kwh['Energy used not in service']
    block_total_kwh['Energy used estimate'] = block_total_kwh['trip_estimated_kwh'] + block_total_kwh['trip_deadhead_drive_kwh'] + block_total_kwh['trip_deadhead_aux_kwh'] + block_total_kwh['block_deadhead_start_kwh'] + block_total_kwh['block_deadhead_end_kwh'] + block_total_kwh['trip_door_kwh'] + block_total_kwh['block_warmup_kwh']
    df = pd.merge(df, block_total_kwh, on='block_id')
    return df, depot_locations


def get_charging_results(energy_res, temperature_f, preconditioning, plug_power_kw):
    # Calculate charging requirements for each block
    block_coverage = energy_res.groupby('block_id').agg({'t_min_of_day': 'first', 't_min_of_day_end': 'last', 'Energy used estimate': 'first', 'block_warmup_kwh_x': 'first'}).sort_values(['block_id', 't_min_of_day'])
    block_coverage['charge_time_min'] = block_coverage['Energy used estimate'] / plug_power_kw * 60
    block_coverage['t_charge_start_min'] = block_coverage['t_min_of_day_end']
    block_coverage['t_charge_end_min'] = block_coverage['t_min_of_day_end'] + block_coverage['charge_time_min']
    block_coverage['t_block_min'] = block_coverage['t_min_of_day_end'] - block_coverage['t_min_of_day']
    block_coverage['t_until_pullout_hr'] = (1441 - block_coverage['t_block_min']) / 60
    block_coverage['min_charge_rate'] = block_coverage['Energy used estimate'] / block_coverage['t_until_pullout_hr']
    block_coverage['t_charge_end_managed_min'] = block_coverage['t_min_of_day_end'] + (block_coverage['t_until_pullout_hr'] * 60)
    temp_differential_f = CABIN_TEMP_F - temperature_f
    if preconditioning and temp_differential_f > 0:
        block_coverage['preconditioning_kwh'] = TOTAL_CABIN_WARMUP_KWH * temp_differential_f
        block_coverage['preconditioning_time_min'] = block_coverage['preconditioning_kwh'] / plug_power_kw * 60
        block_coverage['t_charge_start_min'] = block_coverage['t_charge_start_min'] - block_coverage['preconditioning_time_min']
    # Calculate charging requirements for network
    active_veh = []
    charging_veh = []
    charging_managed_veh = []
    charging_managed_rate = []
    # Check each minute from midnight for the count of active, charging vehicles under managed or unmanaged scenarios
    for min_of_day in np.arange(0, max(block_coverage['t_charge_end_managed_min']), 1):
        x = len(block_coverage[(block_coverage['t_min_of_day'] <= min_of_day) & (block_coverage['t_min_of_day_end'] >= min_of_day)])
        active_veh.append(x)
        x = len(block_coverage[(block_coverage['t_charge_start_min'] <= min_of_day) & (block_coverage['t_charge_end_min'] >= min_of_day)])
        charging_veh.append(x)
        x = len(block_coverage[(block_coverage['t_charge_start_min'] <= min_of_day) & (block_coverage['t_charge_end_managed_min'] >= min_of_day)])
        charging_managed_veh.append(x)
        x = sum(block_coverage[(block_coverage['t_charge_start_min'] <= min_of_day) & (block_coverage['t_charge_end_managed_min'] >= min_of_day)]['min_charge_rate'])
        charging_managed_rate.append(x)
    veh_status = pd.DataFrame({
        "min_of_day": np.arange(0, max(block_coverage['t_charge_end_managed_min']), 1),
        "active_veh": active_veh,
        "charging_veh": charging_veh,
        "charging_managed_veh": charging_managed_veh,
        "charging_managed_rate": charging_managed_rate
    })
    # Reset time to 0-1440
    veh_status.loc[veh_status['min_of_day'] >= 1440, 'min_of_day'] -= 1440
    veh_status['charging_rate'] = veh_status['charging_veh'] * plug_power_kw
    return block_coverage, veh_status