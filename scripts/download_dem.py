import os
from pathlib import Path
import requests

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openbustools import spatial


def download_dem(endpoint, data_folder, grid_bounds, out_file):
    """
    Downloads a Digital Elevation Model (DEM) file from OpenTopography.

    Args:
        endpoint (str): The API endpoint for downloading the DEM.
        data_folder (str): The folder where the downloaded DEM will be saved.
        grid_bounds (list): The bounding coordinates of the grid in the format [min_x, min_y, max_x, max_y].
        out_file (str): The name of the output file.
    """
    print(f"Getting DEM for {data_folder}{out_file} from {endpoint} at OpenTopography")
    try:
        grid_bounds = pd.DataFrame({'x':[grid_bounds[0], grid_bounds[2]], 'y': [grid_bounds[1], grid_bounds[3]]})
        grid_bounds = gpd.GeoDataFrame(grid_bounds, geometry=gpd.points_from_xy(grid_bounds.x, grid_bounds.y), crs=f"EPSG:4326")
        # grid_bounds = gpd.GeoDataFrame(grid_bounds, geometry=gpd.points_from_xy(grid_bounds.x, grid_bounds.y), crs=f"EPSG:{epsg}").to_crs("EPSG:4326")
        params = {
            'API_Key': os.getenv('OPENTOPO_API_KEY'),
            'outputFormat': 'GTiff',
            'south': grid_bounds['geometry'].y[0],
            'north': grid_bounds['geometry'].y[1],
            'west': grid_bounds['geometry'].x[0],
            'east': grid_bounds['geometry'].x[1],
        }
        r = requests.get(endpoint, params=params)
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        with open(Path(data_folder, out_file), "wb") as save_file:
            save_file.write(r.content)
    except:
        print(f"Failure to request data: {r.status_code}")


if __name__ == "__main__":
    # Download KCM
    provider_path = Path('data', 'kcm_spatial')
    provider_path.mkdir(parents=True, exist_ok=True)
    download_dem("https://portal.opentopography.org/API/usgsdem?datasetName=USGS10m", provider_path, [-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], "usgs10m_dem.tif")
    spatial.reproject_raster(provider_path / "usgs10m_dem.tif", provider_path / "usgs10m_dem_32148.tif", 32148)

    # Download ATB
    provider_path = Path('data', 'atb_spatial')
    provider_path.mkdir(parents=True, exist_ok=True)
    download_dem("https://portal.opentopography.org/API/globaldem?demtype=EU_DTM", provider_path, [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], "eudtm30m_dem.tif")
    spatial.reproject_raster(provider_path / "eudtm30m_dem.tif", provider_path / "eudtm30m_dem_32632.tif", 32632)

    # Download Others
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        provider_path = Path('data', 'other_feeds', f"{row['uuid']}_spatial")
        provider_path.mkdir(parents=True, exist_ok=True)
        if row['country_code'] == 'US':
            download_dem("https://portal.opentopography.org/API/usgsdem?datasetName=USGS10m", provider_path, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], "usgs10m_dem.tif")
            spatial.reproject_raster(Path(provider_path, "usgs10m_dem.tif"), Path(provider_path, f"usgs10m_dem_{row['epsg_code']}.tif"), row['epsg_code'])
        elif row['country_code'] in ['FI', 'NO', 'IT', 'ES', 'SE', 'DK', 'DE', 'FR', 'NL', 'BE']:
            download_dem("https://portal.opentopography.org/API/globaldem?demtype=EU_DTM", provider_path, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], "eudtm30m_dem.tif")
            spatial.reproject_raster(Path(provider_path, "eudtm30m_dem.tif"), Path(provider_path, f"eudtm30m_dem_{row['epsg_code']}.tif"), row['epsg_code'])
        else:
            download_dem("https://portal.opentopography.org/API/globaldem?demtype=AW3D30", provider_path, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], "aw3d30m_dem.tif")
            spatial.reproject_raster(Path(provider_path, "aw3d30m_dem.tif"), Path(provider_path, f"aw3d30m_dem_{row['epsg_code']}.tif"), row['epsg_code'])