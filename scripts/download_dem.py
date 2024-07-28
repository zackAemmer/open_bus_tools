import os
from pathlib import Path
import requests

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openbustools import spatial


def download_dem(data_folder, dem_url, grid_bounds, out_file):
    print(f"Getting DEM for {data_folder}{out_file} from {dem_url} at OpenTopography")
    data_folder.mkdir(parents=True, exist_ok=True)
    try:
        grid_bounds = pd.DataFrame({'x':[grid_bounds[0], grid_bounds[2]], 'y': [grid_bounds[1], grid_bounds[3]]})
        grid_bounds = gpd.GeoDataFrame(grid_bounds, geometry=gpd.points_from_xy(grid_bounds.x, grid_bounds.y), crs=f"EPSG:4326")
        params = {
            'API_Key': os.getenv('OPENTOPO_API_KEY'),
            'outputFormat': 'GTiff',
            'south': grid_bounds['geometry'].y[0],
            'north': grid_bounds['geometry'].y[1],
            'west': grid_bounds['geometry'].x[0],
            'east': grid_bounds['geometry'].x[1],
        }
        response = requests.get(dem_url, params=params)
        with open(Path(data_folder, out_file), "wb") as save_file:
            save_file.write(response.content)
    except:
        print(f"ERROR downloading {data_folder} DEM data {response.status_code}")


if __name__ == "__main__":
    download_dem(Path('data','kcm_spatial'), "https://portal.opentopography.org/API/usgsdem?datasetName=USGS10m", [-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], "usgs10m_dem.tif")
    spatial.reproject_raster(Path('data','kcm_spatial','usgs10m_dem.tif'), Path('data','kcm_spatial','usgs10m_dem_32148.tif'), 32148)

    download_dem(Path('data','atb_spatial'), "https://portal.opentopography.org/API/globaldem?demtype=EU_DTM", [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], "eudtm30m_dem.tif")
    spatial.reproject_raster(Path('data','atb_spatial','eudtm30m_dem.tif'), Path('data','atb_spatial','eudtm30m_dem_32632.tif'), 32632)

    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        if row['country_code'] == 'US':
            url, fname = "https://portal.opentopography.org/API/usgsdem?datasetName=USGS10m", "usgs10m_dem"
        elif row['country_code'] in ['FI', 'NO', 'IT', 'ES', 'SE', 'DK', 'DE', 'FR', 'NL', 'BE']:
            url, fname = "https://portal.opentopography.org/API/globaldem?demtype=EU_DTM", "eudtm30m_dem"
        else:
            url, fname = "https://portal.opentopography.org/API/globaldem?demtype=AW3D30", "aw3d30m_dem"
        download_dem(Path('data','other_feeds',f"{row['uuid']}_spatial"), url, [row['min_lon'],row['min_lat'],row['max_lon'],row['max_lat']], f"{fname}.tif")
        spatial.reproject_raster(Path('data','other_feeds',f"{row['uuid']}_spatial",f"{fname}.tif"), Path('data','other_feeds',f"{row['uuid']}_spatial",f"{fname}_{row['epsg_code']}.tif"), row['epsg_code'])