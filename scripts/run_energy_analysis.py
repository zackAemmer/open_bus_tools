import pickle
import sys
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()
import geopandas as gpd
import importlib
import copy
import logging
import contextily as cx
import gtfs_kit as gk
import fastsim as fsim
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import lightning.pytorch as pl
import rasterio as rio
import seaborn as sns
from rasterio.plot import show
import seaborn as sns
import shapely
import statsmodels.api as sm
from torch.utils.data import DataLoader

from openbustools.traveltime.models import rnn
from openbustools import plotting, spatial, standardfeeds
from openbustools.traveltime import data_loader, model_utils
from openbustools.drivecycle import trajectory, busnetwork
from openbustools.drivecycle.physics import conditions, energy, vehicle


def build_trajectories(**kwargs):
    logger.debug(f"BUILDING TRAJECTORIES: {kwargs['save_dir']}")
    # Get most recent static feed
    latest_static_file = standardfeeds.latest_available_static(kwargs['target_day'], kwargs['static_dir'])
    latest_static_file = kwargs['static_dir'] / latest_static_file
    # Build a trajectory for each trip in the static feed
    trajectories = busnetwork.get_trajectories(latest_static_file, target_day=kwargs['target_day'], epsg=kwargs['epsg'], coord_ref_center=kwargs['coord_ref_center'], dem_file=kwargs['dem_file'], min_traj_n=7)
    # Create results folder if it doesn't exist
    save_dir = kwargs['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save trajectories
    file = open(save_dir / "trajectories.pkl", "wb")
    pickle.dump(trajectories, file)
    file.close()


def predict_times(**kwargs):
    logger.debug(f"PREDICTING TIMES: {kwargs['save_dir']}")
    # Load trajectories
    file = open(kwargs['load_dir'] / "trajectories.pkl", "rb")
    trajectories = pickle.load(file)
    file.close()
    # Get most recent tuned model
    latest_version = str(sorted([int(x.split('_')[1]) for x in os.listdir(kwargs['model_dir'])])[-1])
    latest_ckpt = sorted(os.listdir(kwargs['model_dir'] / f"version_{latest_version}" / "checkpoints"))[-1]
    model = rnn.GRU.load_from_checkpoint(kwargs['model_dir'] / f"version_{latest_version}" / "checkpoints" / latest_ckpt, strict=False).eval()
    # Predict travel times for each trajectory
    busnetwork.update_travel_times(trajectories, model)
    model_errors = [(t.gdf['pred_speed_m_s']<0).any() for t in trajectories]
    trajectories = list(np.array(trajectories)[~np.array(model_errors)])
    logger.debug(f"Dropping {sum(model_errors)} trajectories with negative predicted speeds")
    model_errors = [(t.gdf['pred_speed_m_s']>35).any() for t in trajectories]
    trajectories = list(np.array(trajectories)[~np.array(model_errors)])
    logger.debug(f"Dropping {sum(model_errors)} trajectories with 35m/s+ predicted speeds")
    # Create results folder if it doesn't exist
    save_dir = kwargs['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save updated trajectories
    file = open(save_dir / "trajectories_updated.pkl", "wb")
    pickle.dump(trajectories, file)
    file.close()


def calculate_cycle_energy(**kwargs):
    logger.debug(f"CALCULATING CYCLE ENERGIES: {kwargs['save_dir']}")
    # Load updated trajectories
    file = open(kwargs['load_dir'] / "trajectories_updated.pkl", "rb")
    trajectories = pickle.load(file)
    file.close()
    # Turn each trajectory to drive cycle, calculate energy consumption
    veh = fsim.vehicle.Vehicle.from_vehdb(kwargs['veh_num'], veh_file=kwargs['veh_file'])
    cycles = [busnetwork.get_trajectory_energy(traj, veh, kwargs['sensitivity_params']['passenger_load'], kwargs['sensitivity_params']['aux_power_kw'], kwargs['sensitivity_params']['acc_dec_factor']) for traj in trajectories]
    # Create results folder if it doesn't exist
    save_dir = kwargs['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save updated trajectories
    file = open(save_dir / "trajectories_updated.pkl", "wb")
    pickle.dump(trajectories, file)
    file.close()
    # Save cycles
    file = open(save_dir / "cycles.pkl", "wb")
    pickle.dump(cycles, file)
    file.close()


def postprocess_network_energy(**kwargs):
    logger.debug(f"POSTPROCESSING NETWORK ENERGY: {kwargs['save_dir']}")
    # Load updated trajectories
    file = open(kwargs['load_dir'] / "trajectories_updated.pkl", "rb")
    trajectories = pickle.load(file)
    file.close()
    # Load cycles
    file = open(kwargs['load_dir'] / "cycles.pkl", "rb")
    cycles = pickle.load(file)
    file.close()
    # Postprocess each network
    network_energy, depot_locations = busnetwork.get_energy_results(
        trajectories,
        cycles,
        n_depots=max([1, int(kwargs['network_area_sqkm'] * kwargs['sensitivity_params']['depot_density_per_sqkm'])]),
        deadhead_consumption_kwh_mi=kwargs['sensitivity_params']['deadhead_consumption_kwh_mi'],
        deadhead_aux_kw=kwargs['sensitivity_params']['aux_power_kw'],
        temperature_f=kwargs['sensitivity_params']['temperature_f'],
        door_open_time_s=kwargs['sensitivity_params']['door_open_time_s']
    )
    # Create results folder if it doesn't exist
    save_dir = kwargs['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save postprocessed results
    file = open(save_dir / "network_energy.pkl", "wb")
    pickle.dump(network_energy, file)
    file.close()
    file = open(save_dir / "depot_locations.pkl", "wb")
    pickle.dump(depot_locations, file)
    file.close()


def postprocess_network_charging(**kwargs):
    logger.debug(f"POSTPROCESSING NETWORK CHARGING: {kwargs['save_dir']}")
    # Load block energy
    file = open(kwargs['load_dir'] / "network_energy.pkl", "rb")
    network_energy = pickle.load(file)
    file.close()
    # Postprocess each network
    network_charging, veh_status = busnetwork.get_charging_results(
        network_energy,
        plug_power_kw=kwargs['sensitivity_params']['depot_plug_power_kw'],
    )
    # Create results folder if it doesn't exist
    save_dir = kwargs['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save postprocessed results
    file = open(save_dir / "network_charging.pkl", "wb")
    pickle.dump(network_charging, file)
    file.close()
    file = open(save_dir / "veh_status.pkl", "wb")
    pickle.dump(veh_status, file)
    file.close()


def postprocess_network_sensitivity(**kwargs):
    logger.debug(f"POSTPROCESSING NETWORK SENSITIVITY: {kwargs['save_dir']}")
    network_sensitivity = busnetwork.get_sensitivity_results(kwargs['load_dir'])
    network_sensitivity['provider'] = kwargs['provider']
    # Create results folder if it doesn't exist
    save_dir = kwargs['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save postprocessed results
    file = open(save_dir / "network_sensitivity.pkl", "wb")
    pickle.dump(network_sensitivity, file)
    file.close()


def sensitivity_analysis(baseline_only=False, **kwargs):
    logger.debug(f"SENSITIVITY ANALYSIS: {kwargs['save_dir']}")
    logger.debug(f"Running sensitivity for baseline")
    # Set the sensitivity parameter to be tested; keep baseline for others
    run_sensitivity_params = copy.deepcopy(kwargs['sensitivity_params'])
    calculate_cycle_energy(
        load_dir=kwargs['load_dir'],
        save_dir=Path(kwargs['save_dir'], f"baseline"),
        veh_file=kwargs['veh_file'],
        veh_num=kwargs['veh_num'],
        sensitivity_params=run_sensitivity_params,
    )
    postprocess_network_energy(
        load_dir=Path(kwargs['save_dir'], f"baseline"),
        save_dir=Path(kwargs['save_dir'], f"baseline"),
        network_area_sqkm=kwargs['network_area_sqkm'],
        sensitivity_params=run_sensitivity_params,
    )
    postprocess_network_charging(
        load_dir=Path(kwargs['save_dir'], f"baseline"),
        save_dir=Path(kwargs['save_dir'], f"baseline"),
        sensitivity_params=run_sensitivity_params,
    )
    # Save the sensitivity parameters
    file = open(Path(kwargs['save_dir'], f"baseline", "sensitivity_params.pkl"), "wb")
    pickle.dump(run_sensitivity_params, file)
    file.close()
    if not baseline_only:
        # Run other param values
        for param, values in kwargs['sensitivity_ranges'].items():
            for i, v in enumerate(values):
                logger.debug(f"Running sensitivity for {param}={v}")
                # Set the sensitivity parameter to be tested; keep baseline for others
                run_sensitivity_params = copy.deepcopy(kwargs['sensitivity_params'])
                run_sensitivity_params[param] = v
                calculate_cycle_energy(
                    load_dir=kwargs['load_dir'],
                    save_dir=Path(kwargs['save_dir'], f"{param}-{i}"),
                    veh_file=kwargs['veh_file'],
                    veh_num=kwargs['veh_num'],
                    sensitivity_params=run_sensitivity_params,
                )
                postprocess_network_energy(
                    load_dir=Path(kwargs['save_dir'], f"{param}-{i}"),
                    save_dir=Path(kwargs['save_dir'], f"{param}-{i}"),
                    network_area_sqkm=kwargs['network_area_sqkm'],
                    sensitivity_params=run_sensitivity_params,
                )
                postprocess_network_charging(
                    load_dir=Path(kwargs['save_dir'], f"{param}-{i}"),
                    save_dir=Path(kwargs['save_dir'], f"{param}-{i}"),
                    sensitivity_params=run_sensitivity_params,
                )
                # Save the sensitivity parameters
                file = open(Path(kwargs['save_dir'], f"{param}-{i}", "sensitivity_params.pkl"), "wb")
                pickle.dump(run_sensitivity_params, file)
                file.close()


if __name__=="__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('run_energy_analysis')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    n_sensitivity = 5
    sensitivity_ranges = {
        # Modify cycle parameters for FastSIM
        "acc_dec_factor": np.linspace(0.1, 5.0, n_sensitivity),
        # Modify vehicle parameters for FastSIM
        "passenger_load": np.linspace(1, 82, n_sensitivity),
        "aux_power_kw": np.linspace(1, 40, n_sensitivity),
        # Modify total energy postprocessing
        "deadhead_consumption_kwh_mi": np.linspace(2.0, 5.0, n_sensitivity),
        "depot_density_per_sqkm": np.linspace(0.0002, 0.002, n_sensitivity),
        "door_open_time_s": np.linspace(10, 120, n_sensitivity),
        # Modify both total energy and charging postprocessing
        "temperature_f": np.linspace(20, 90, n_sensitivity),
        # Modify charging postprocessing
        "depot_plug_power_kw": np.linspace(50, 350, n_sensitivity),
    }
    sensitivity_baseline_params = {
        "acc_dec_factor": 1.0,
        "passenger_load": 41,
        "aux_power_kw": 20,
        "deadhead_consumption_kwh_mi": 2.77,
        "depot_density_per_sqkm": 0.0012,
        "door_open_time_s": 30,
        "temperature_f": 46,
        "depot_plug_power_kw": 50,
    }

    # # KCM
    # build_trajectories(
    #     load_dir=Path("results","energy","kcm"),
    #     save_dir=Path("results","energy","kcm"),
    #     static_dir=Path("data","kcm_static"),
    #     dem_file=Path("data","kcm_spatial","usgs10m_dem_32148.tif"),
    #     epsg=32148,
    #     coord_ref_center=[386910,69022],
    #     target_day="2023_12_01",
    # )
    # predict_times(
    #     load_dir=Path("results","energy","kcm"),
    #     save_dir=Path("results","energy","kcm"),
    #     model_dir=Path("logs","kcm","GRU-4"),
    # )
    # sensitivity_analysis(
    #     # baseline_only=True,
    #     load_dir=Path("results","energy","kcm"),
    #     save_dir=Path("results","energy","kcm","sensitivity"),
    #     veh_file=Path("data","FASTSim_py_veh_db.csv"),
    #     veh_num=63,
    #     network_area_sqkm=5530,
    #     sensitivity_params=sensitivity_baseline_params,
    #     sensitivity_ranges=sensitivity_ranges,
    # )
    # postprocess_network_sensitivity(
    #     load_dir=Path('results', 'energy', 'kcm', "sensitivity"),
    #     save_dir=Path('results', 'energy', 'kcm'),
    #     provider='kcm'
    # )

    # Other feeds
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    skip_uuids = [
        "b555d2ea-4c39-4e5f-b32d-2606984bd8ac",
        "b6215bd6-c5c1-4a41-b134-0170c097e83f",
        "b1086c64-bb1a-4639-a283-9a8375f2ad4a",
        "b2754f7b-6ece-4f4f-a62e-44ae3b1b08ef",
        "afe2a7f1-1b7b-4cc9-9704-c67afc90f018",
        "e718826c-e736-4c0b-a51c-5a33d110fe76",
        "499799a8-9474-4525-83d1-a160bf7b1fe7",
        "16d0eac0-d3ab-4072-96f0-b508a3eea582",
        "6e31ba64-3fe4-4ea9-8889-7a069c1cc7f2"]
    for i, row in cleaned_sources.iterrows():
        if row['uuid'] in skip_uuids:
            continue
        if "sensitivity" in [x.name for x in Path('results','energy',row['uuid']).glob("*")]:
            if "depot_plug_power_kw-4" in [x.name for x in Path('results','energy',row['uuid'],"sensitivity").glob("*")]:
                continue
        # build_trajectories(
        #     load_dir=Path('results', 'energy', f"{row['uuid']}"),
        #     save_dir=Path('results', 'energy', f"{row['uuid']}"),
        #     static_dir=Path('data', 'other_feeds', f"{row['uuid']}_static"),
        #     dem_file=[x for x in Path('data', 'other_feeds', f"{row['uuid']}_spatial").glob(f"*_{row['epsg_code']}.tif")][0],
        #     epsg=row['epsg_code'],
        #     coord_ref_center=[row['coord_ref_x'], row['coord_ref_y']],
        #     target_day="2024_04_03"
        # )
        # predict_times(
        #     load_dir=Path('results', 'energy', f"{row['uuid']}"),
        #     save_dir=Path('results', 'energy', f"{row['uuid']}"),
        #     model_dir=Path('logs','other_feeds', f"{row['uuid']}", "lightning_logs")
        # )
        sensitivity_analysis(
            load_dir=Path('results', 'energy', f"{row['uuid']}"),
            save_dir=Path('results', 'energy', f"{row['uuid']}", "sensitivity"),
            veh_file=Path("data","FASTSim_py_veh_db.csv"),
            veh_num=63,
            network_area_sqkm=int(spatial.bbox_area(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat'])),
            sensitivity_params=sensitivity_baseline_params,
            sensitivity_ranges=sensitivity_ranges,
        )
        postprocess_network_sensitivity(
            load_dir=Path('results', 'energy', f"{row['uuid']}", "sensitivity"),
            save_dir=Path('results', 'energy', f"{row['uuid']}"),
            provider=row['provider']
        )