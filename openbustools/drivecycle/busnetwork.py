import pickle
import copy
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
CABIN_VOLUME_FT3 = 10*40*8
# Multiply by temperature differential in F to get kWh:
TOTAL_CABIN_WARMUP_KWH = CABIN_VOLUME_FT3*AIR_DENSITY_LB_FT3*AIR_SPECIFIC_HEAT_BTU_LBDEG/3412

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


def get_trajectories(static_dir, target_day, epsg, coord_ref_center, dem_file, point_sep_m=300, min_traj_n=3):
    # Load static feed restricted to target day
    static_feed = gk.read_feed(static_dir, dist_units="km").restrict_to_dates([str.replace(target_day,'_','')])
    # Some feeds have not updated the calendar in a while
    if len(static_feed.trips)==0:
        static_feed = gk.read_feed(static_dir, dist_units="km")
    # Filter trips to active day of week
    t_day_of_week = datetime.strptime(target_day, "%Y_%m_%d").weekday()
    # Calendar conditional on being multiple service IDs
    if len(static_feed.trips['service_id'].unique()) > 1:
        if static_feed.calendar is None:
            active_service_ids = [static_feed.trips.groupby('service_id', as_index=False).count().sort_values('trip_id')['service_id'].iloc[-1]]
        # Regardless some are still broken
        else:
            active_service_ids = static_feed.calendar[static_feed.calendar.iloc[:,t_day_of_week+1]==1]['service_id'].to_numpy()
    else:
        active_service_ids = static_feed.trips['service_id'].unique()
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
        shapeid = df['shape_id'].iloc[0]
        if shapeid not in route_shape_points.keys():
            print(f"Skipping trip {trip_id} with no shape points")
            continue
        else:
            shape_df = route_shape_points[shapeid]
        if len(shape_df) <= 1:
            print(f"Skipping trip {trip_id} with only {len(shape_df)} points")
            continue
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


def get_trajectory_energy(traj, veh, passenger_load, aux_power_kw, acc_dec_factor):
    # Incorporate acceleration/deceleration parameter
    filtered_mps = spatial.apply_peak_filter(traj.gdf['pred_speed_m_s'].to_numpy(), window_len=3, acc_scalar=acc_dec_factor, dec_scalar=acc_dec_factor, clip_min=0, clip_max=35)
    filtered_mps = spatial.apply_sg_filter(filtered_mps, clip_min=0, clip_max=35)
    cycle_pred = {
        "cycGrade": np.clip(spatial.divide_fwd_back_fill(np.diff(traj.gdf['calc_elev_m'], prepend=traj.gdf['calc_elev_m'].iloc[0]), traj.gdf['calc_dist_m']), -0.15, 0.15),
        "mps": filtered_mps,
        "time_s": traj.gdf['cumul_time_s'].to_numpy(),
        "road_type": np.zeros(len(traj.gdf))
    }
    cycle_pred = fsim.cycle.Cycle.from_dict(fsim.cycle.resample(cycle_pred, new_dt=1)).to_rust()
    # Incorporate static aux power parameter
    veh_mod = copy.deepcopy(veh)
    veh_mod.aux_kw = aux_power_kw
    # Incorporate passenger load parameter
    veh_mod.veh_override_kg = veh_mod.veh_override_kg + (150 * passenger_load * 0.453592)
    veh_mod = veh_mod.to_rust()
    sim_drive_pred = fsim.simdrive.RustSimDrive(cycle_pred, veh_mod)
    sim_drive_pred.sim_drive()
    sim_drive_pred = fsim.simdrive.copy_sim_drive(sim_drive_pred, 'python')
    # Minimize space taken by cycles as much as possible
    res = {
        "electric_kwh_per_mi": np.array(copy.deepcopy(sim_drive_pred.electric_kwh_per_mi)),
        "ess_kw_out_ach": np.array(copy.deepcopy(sim_drive_pred.ess_kw_out_ach)),
        # "accel_kw": np.array(copy.deepcopy(sim_drive_pred.accel_kw)),
        # "rr_kw": np.array(copy.deepcopy(sim_drive_pred.rr_kw)),
        # "ess_loss_kw": np.array(copy.deepcopy(sim_drive_pred.ess_loss_kw)),
        # "ascent_kw": np.array(copy.deepcopy(sim_drive_pred.ascent_kw)),
        # "drag_kw": np.array(copy.deepcopy(sim_drive_pred.drag_kw)),
        # "aux_in_kw": np.array(copy.deepcopy(sim_drive_pred.aux_in_kw)),
    }
    del(cycle_pred)
    del(veh_mod)
    return res


def get_energy_results(network_trajs, network_cycles, n_depots, deadhead_consumption_kwh_mi, deadhead_aux_kw, temperature_f, door_open_time_s):
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
    trip_consumption = [float(cycle['electric_kwh_per_mi']) for cycle in network_cycles]
    trip_distance = [int(np.sum(t.gdf['calc_dist_m'])) for t in network_trajs]
    start_loc = [t.gdf.iloc[0].geometry for t in network_trajs]
    end_loc = [t.gdf.iloc[-1].geometry for t in network_trajs]
    t_min_of_day = [t.traj_attr['t_min_of_day'] for t in network_trajs]
    t_min_of_day_end = [t.traj_attr['t_min_of_day_end'] for t in network_trajs]

    # Energy use and regeneration for drive cycle and estimate of net energy for block
    total_cyc_regen_kwh = [np.sum(cycle['ess_kw_out_ach'][cycle['ess_kw_out_ach']<0]) / 3600 for cycle in network_cycles]
    total_cyc_energy_kwh = [cycle['electric_kwh_per_mi'] * (trip_distance[i] / 1000 / 1.609) for i, cycle in enumerate(network_cycles)]

    # Create a dataframe with rows for each trip in the network on the target day
    df = pd.DataFrame({
        'network_id': network_id,
        'block_id': block_ids,
        'trip_id': trip_ids,
        't_min_of_day': t_min_of_day,
        't_min_of_day_end': t_min_of_day_end,
        'consumption_kwh_mi': trip_consumption,
        'trip_dist_m': trip_distance,
        'start_loc': start_loc,
        'end_loc': end_loc,
        'num_trips': num_trips,
        'num_blocks': num_blocks,
        'cyc_regen_kwh': total_cyc_regen_kwh,
        'cyc_energy_kwh': total_cyc_energy_kwh
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
    df['trip_deadhead_drive_kwh'] = df['trip_deadhead_dist_m'] / 1000 / 1.609 * deadhead_consumption_kwh_mi
    # Assume if downtime is greater than 1 hour, the bus is turned off
    df['trip_deadhead_aux_kwh'] = np.clip(df['t_min_down'], a_min=0, a_max=60) * 60 * deadhead_aux_kw / 3600
    df['trip_deadhead_kwh'] = df['trip_deadhead_drive_kwh'] + df['trip_deadhead_aux_kwh']

    # Calculate energy use for HVAC related to door opening
    temp_differential_f = CABIN_TEMP_F - temperature_f
    if temp_differential_f == 0:
        tot_door_kwh = np.array([0.0 for t in network_trajs])
    else:
        # Constants
        # ~140s for all bus air to cycle
        door_area_ft2 = 8*4*2*0.5
        wind_speed_ft_s = 0.5
        air_density_lb_ft3 = 0.075
        air_specific_heat_btu_lbdeg = 0.24
        heat_eff = 1.0
        cool_eff = 1.0
        # Energy from air loss
        Q_air_ft3_s = door_area_ft2 * wind_speed_ft_s
        Q_air_lb_s = Q_air_ft3_s * air_density_lb_ft3
        Q_air_btu_s = Q_air_lb_s * temp_differential_f * air_specific_heat_btu_lbdeg
        # Outdoor temperature is less than cabin
        if temp_differential_f > 0:
            hvac_btu_s = Q_air_btu_s / heat_eff
        # Outdoor temperature is greater than cabin
        else:
            hvac_btu_s = Q_air_btu_s / cool_eff
        tot_door_times_s = np.array([t.traj_attr['stop_count'] * door_open_time_s for t in network_trajs])
        tot_door_btu = tot_door_times_s * hvac_btu_s
        tot_door_kwh = tot_door_btu / 3412
    df['trip_door_kwh'] = tot_door_kwh

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
    depot_assignments['block_deadhead_dist_m'] = depot_assignments['block_deadhead_start_dist_m'] + depot_assignments['block_deadhead_end_dist_m']
    depot_assignments['block_deadhead_start_kwh'] = depot_assignments['block_deadhead_start_dist_m'] / 1000 / 1.609 * deadhead_consumption_kwh_mi
    depot_assignments['block_deadhead_end_kwh'] = depot_assignments['block_deadhead_end_dist_m'] / 1000 / 1.609 * deadhead_consumption_kwh_mi
    depot_assignments['block_deadhead_kwh'] = depot_assignments['block_deadhead_start_kwh'] + depot_assignments['block_deadhead_end_kwh']
    depot_assignments = depot_assignments[['block_id','depot_id','block_deadhead_kwh','block_deadhead_dist_m']].copy()
    df = pd.merge(df, depot_assignments, on="block_id")

    # Block-level summary metrics
    block_summary = df.groupby('block_id').agg({
        # Total block energy
        'cyc_energy_kwh': 'sum',
        'trip_door_kwh': 'sum',
        'trip_deadhead_kwh': 'sum',
        'block_deadhead_kwh': 'first',
        # Total block regen (no deadhead)
        'cyc_regen_kwh': 'sum',
        # Total block distance
        'trip_dist_m': 'sum',
        'trip_deadhead_dist_m': 'sum',
        'block_deadhead_dist_m': 'first',
    }).reset_index()
    block_summary['block_net_energy_kwh'] = block_summary['cyc_energy_kwh'] + block_summary['trip_door_kwh'] + block_summary['trip_deadhead_kwh'] + block_summary['block_deadhead_kwh']
    block_summary['block_cyc_regen_kwh'] = block_summary['cyc_regen_kwh']
    block_summary['block_dist_mi'] = (block_summary['trip_dist_m'] + block_summary['trip_deadhead_dist_m'] + block_summary['block_deadhead_dist_m']) / 1000 / 1.609
    block_summary['block_consumption_kwh_mi'] = block_summary['block_net_energy_kwh'] / block_summary['block_dist_mi']
    block_summary = block_summary[['block_id','block_net_energy_kwh','block_cyc_regen_kwh','block_dist_mi','block_consumption_kwh_mi']].copy()
    df = pd.merge(df, block_summary, on="block_id")
    return df, depot_locations


def get_charging_results(energy_res, plug_power_kw):
    # Calculate charging requirements for each block
    block_coverage = energy_res.groupby('block_id').agg({'t_min_of_day': 'first', 't_min_of_day_end': 'last', 'block_net_energy_kwh': 'first'}).sort_values(['block_id', 't_min_of_day'])
    block_coverage['charge_time_min'] = block_coverage['block_net_energy_kwh'] / plug_power_kw * 60
    block_coverage['t_charge_start_min'] = block_coverage['t_min_of_day_end']
    block_coverage['t_charge_end_min'] = block_coverage['t_min_of_day_end'] + block_coverage['charge_time_min']
    block_coverage['t_block_min'] = block_coverage['t_min_of_day_end'] - block_coverage['t_min_of_day']
    block_coverage['t_until_pullout_min'] = (1441 - block_coverage['t_block_min'])
    block_coverage['min_charge_rate'] = block_coverage['block_net_energy_kwh'] / (block_coverage['t_until_pullout_min'] / 60)
    block_coverage['plug_power_kw'] = plug_power_kw
    block_coverage = block_coverage.sort_values('t_min_of_day')

    # Calculate charging status by time of day
    t_mins = np.arange(0, 1440+1440*2)
    veh_status_df = pd.DataFrame({
        't_min_of_day': t_mins,
        'tot_veh_active': [len(block_coverage[(block_coverage['t_min_of_day']<=t) & (block_coverage['t_min_of_day_end']>=t)]) for t in t_mins],
        'tot_veh_inactive': [len(block_coverage[(block_coverage['t_min_of_day']>t) | (block_coverage['t_min_of_day_end']<t)]) for t in t_mins],
        'tot_veh_charging': [len(block_coverage[(block_coverage['t_charge_start_min']<=t) & (block_coverage['t_charge_end_min']>=t)]) for t in t_mins],
        'tot_veh_arriving': [len(block_coverage[block_coverage['t_min_of_day_end']==t]) for t in t_mins],
        'tot_veh_departing': [len(block_coverage[block_coverage['t_min_of_day']==t]) for t in t_mins],
        'tot_energy_arriving': [block_coverage[block_coverage['t_min_of_day_end']==t]['block_net_energy_kwh'].sum() for t in t_mins],
        'tot_energy_departing': [block_coverage[block_coverage['t_min_of_day']==t]['block_net_energy_kwh'].sum() for t in t_mins],
        'tot_power': [block_coverage[(block_coverage['t_charge_start_min']<=t) & (block_coverage['t_charge_end_min']>=t)]['plug_power_kw'].sum() for t in t_mins],
    })
    # Reset time to 0-1440
    veh_status_df.loc[veh_status_df['t_min_of_day'] >= 2*1440, 't_min_of_day'] -= 2*1440
    veh_status_df.loc[veh_status_df['t_min_of_day'] >= 1440, 't_min_of_day'] -= 1440
    veh_status_df = veh_status_df.groupby('t_min_of_day', as_index=False).agg({
        'tot_veh_active': 'sum',
        'tot_veh_inactive': 'min',
        'tot_veh_charging': 'sum',
        'tot_veh_arriving': 'sum',
        'tot_veh_departing': 'sum',
        'tot_energy_arriving': 'sum',
        'tot_energy_departing': 'sum',
        'tot_power': 'sum'
    }).sort_values('t_min_of_day')

    # Calculate kW needed to meet energy demand, based on veh-hrs available at base
    total_daily_energy = block_coverage.groupby('block_id').first()['block_net_energy_kwh'].sum()
    total_daily_charge_veh_hr = veh_status_df['tot_veh_inactive'].sum() / 60
    block_coverage['min_charge_rate_managed'] = total_daily_energy / total_daily_charge_veh_hr

    return block_coverage, veh_status_df


def get_sensitivity_results(sensitivity_dir, bus_battery_capacity_kwh=466):
    all_res = []
    # Sensitivity directories
    sensitivity_files = sensitivity_dir.glob("*")
    sensitivity_files = [f for f in sensitivity_files if f.is_dir()]
    # Calculate comparison metrics for each sensitivity run
    for file in sensitivity_files:
        network_energy = pd.read_pickle(Path(file, "network_energy.pkl"))
        network_charging = pd.read_pickle(Path(file, "network_charging.pkl"))
        veh_status = pd.read_pickle(Path(file, "veh_status.pkl"))
        with open(Path(file, "sensitivity_params.pkl"), "rb") as pkl_file:
            params = pickle.load(pkl_file)
        res = pd.DataFrame({
            'file': file.name,
            'Number of Trips': len(network_energy),
            'Number of Blocks': len(network_charging),
            'Avg. Block Distance (mi)': network_energy.groupby('block_id').first()['block_dist_mi'].mean(),
            'Avg. Block Duration (min)': network_charging['t_block_min'].mean(),
            'Avg. Block Energy (kWh)': network_energy.groupby('block_id').first()['block_net_energy_kwh'].mean(),
            'Avg. Block Consumption (kWh/mi)': network_energy.groupby('block_id').first()['block_consumption_kwh_mi'].mean(),
            'Avg. Trip Consumption (kWh/mi)': network_energy['consumption_kwh_mi'].mean(),
            'Proportion Blocks Meeting Energy': len(network_charging[network_charging['block_net_energy_kwh'] < bus_battery_capacity_kwh]) / len(network_charging),
            'Proportion Blocks Meeting Charging': len(network_charging[network_charging['t_until_pullout_min'] > network_charging['charge_time_min']]) / len(network_charging),
            'Avg. Charge Time (min)': network_charging['charge_time_min'].mean(),
            'Min Charge Rate (kW)': network_charging['min_charge_rate_managed'].min(),
            'Peak 15min Power (kW)': veh_status['tot_power'].max(),
            'Avg. 15min Power (kW)': veh_status['tot_power'].mean(),
            'Battery Capacity 10% (kWh)': np.percentile(network_energy.groupby('block_id').first()['block_net_energy_kwh'], 10),
            'Battery Capacity 95% (kWh)': np.percentile(network_energy.groupby('block_id').first()['block_net_energy_kwh'], 95),
            'Charger Power 10% (kW)': np.percentile(network_charging['min_charge_rate'], 10),
            'Charger Power 95% (kW)': np.percentile(network_charging['min_charge_rate'], 95),
        }, index=[0])
        all_res.append(res)
    all_res = pd.concat(all_res)
    all_res = all_res.melt(id_vars='file', var_name='metric', value_name='value')
    all_res['sensitivity_parameter'] = all_res['file'].str.split('-').str[0]
    all_res['sensitivity_parameter'] = all_res['sensitivity_parameter'].replace({
        'acc_dec_factor': 'Acc./Dec. Factor',
        'passenger_load': 'Passenger Load',
        'door_open_time_s': 'Door Open Time',
        'depot_density_per_sqkm': 'Depot Density',
        'deadhead_consumption_kwh_mi': 'Deadhead Consumption',
        'temperature_f': 'Temperature',
        'aux_power_kw': 'Aux Power',
        'depot_plug_power_kw': 'Plug Power',
    })
    return all_res