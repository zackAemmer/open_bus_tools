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


def network_energy_analysis(**kwargs):
    logger.debug(f"RUNNING ENERGY ANALYSIS: {kwargs['network_name']}")

    # Get most recent tuned model
    latest_version = str(sorted([int(x.split('_')[1]) for x in os.listdir(kwargs['model_dir'])])[-1])
    latest_ckpt = sorted(os.listdir(kwargs['model_dir'] / f"version_{latest_version}" / "checkpoints"))[-1]
    model = rnn.GRU.load_from_checkpoint(kwargs['model_dir'] / f"version_{latest_version}" / "checkpoints" / latest_ckpt, strict=False).eval()

    # Get most recent static feed
    latest_static_file = standardfeeds.latest_available_static(kwargs['target_day'], kwargs['static_dir'])
    latest_static_file = kwargs['static_dir'] / latest_static_file

    save_dir = Path("results","energy",kwargs['network_name'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    try:
        if not kwargs['load_trajs']:
            logger.debug(f"BUILDING TRAJECTORIES: {kwargs['network_name']}")
            trajectories = busnetwork.get_trajectories(latest_static_file, target_day=kwargs['target_day'], epsg=kwargs['epsg'], coord_ref_center=kwargs['coord_ref_center'], dem_file=kwargs['dem_file'])
            file = open(save_dir / "trajectories.pkl", "wb")
            pickle.dump(trajectories, file)
            file.close()
        else:
            logger.debug(f"LOADING TRAJECTORIES: {kwargs['network_name']}")
            file = open(save_dir / "trajectories.pkl", "rb")
            trajectories = pickle.load(file)
            file.close()
        if not kwargs['load_times']:
            logger.debug(f"PREDICTING TIMES: {kwargs['network_name']}")
            busnetwork.update_travel_times(trajectories, model)
            model_errors = [(t.gdf['pred_speed_m_s']<0).any() for t in trajectories]
            trajectories = list(np.array(trajectories)[~np.array(model_errors)])
            logger.debug(f"Dropping {sum(model_errors)} trajectories with negative predicted speeds")
            model_errors = [(t.gdf['pred_speed_m_s']>35).any() for t in trajectories]
            trajectories = list(np.array(trajectories)[~np.array(model_errors)])
            logger.debug(f"Dropping {sum(model_errors)} trajectories with 35m/s+ predicted speeds")
            file = open(save_dir / "trajectories_updated.pkl", "wb")
            pickle.dump(trajectories, file)
            file.close()
        else:
            logger.debug(f"LOADING TRAJECTORIES WITH TIMES: {kwargs['network_name']}")
            file = open(save_dir / "trajectories_updated.pkl", "rb")
            trajectories = pickle.load(file)
            file.close()
        if not kwargs['load_cycles']:
            logger.debug(f"CALCULATING DRIVE CYCLE ENERGY: {kwargs['network_name']}")
            cycles = [busnetwork.get_trajectory_energy(traj, kwargs['veh_file']) for traj in trajectories]
            file = open(save_dir / "cycles.pkl", "wb")
            pickle.dump(cycles, file)
            file.close()
        else:
            logger.debug(f"LOADING DRIVE CYCLE ENERGY: {kwargs['network_name']}")
            file = open(save_dir / "cycles.pkl", "rb")
            cycles = pickle.load(file)
            file.close()
    except Exception as e:
        logger.debug(f"ERROR: {kwargs['network_name']}")
        logger.debug(e)


if __name__=="__main__":
    logger = logging.getLogger('run_energy_analysis')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    network_energy_analysis(
        network_name="mix",
        static_dir=Path("data","kcm_static"),
        model_dir=Path("logs","mix","GRU-0"),
        dem_file = Path("data","kcm_spatial","usgs10m_dem_32148.tif"),
        veh_file = Path("data","FASTSim_py_veh_db.csv"),
        epsg=32148,
        coord_ref_center=[386910,69022],
        target_day="2023_12_01",
        load_trajs=True,
        load_times=False,
        load_cycles=False
    )

    # cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    # for i, row in cleaned_sources.iloc[:].iterrows():
    #     network_energy_analysis(
    #         network_name=row['uuid'],
    #         static_dir=Path('data', 'other_feeds', f"{row['uuid']}_static"),
    #         model_dir=Path('logs','other_feeds', f"{row['uuid']}", "lightning_logs"),
    #         dem_file=[x for x in Path('data', 'other_feeds', f"{row['uuid']}_spatial").glob(f"*_{row['epsg_code']}.tif")][0],
    #         veh_file = Path("data","FASTSim_py_veh_db.csv"),
    #         epsg=row['epsg_code'],
    #         coord_ref_center=[row['coord_ref_x'], row['coord_ref_y']],
    #         target_day="2024_01_03",
    #         start_step=0
    #     )