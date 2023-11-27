import os
import requests

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


def download_dem(data_folder, grid_bounds, epsg, out_file):
    print(f"Getting DEM for {data_folder}{out_file} from OpenTopography")
    try:
        grid_bounds = pd.DataFrame({'x':[grid_bounds[0], grid_bounds[2]], 'y': [grid_bounds[1], grid_bounds[3]]})
        grid_bounds = gpd.GeoDataFrame(grid_bounds, geometry=gpd.points_from_xy(grid_bounds.x, grid_bounds.y), crs=f"EPSG:{epsg}").to_crs("EPSG:4326")
        params = {
            'API_Key': os.getenv('OPENTOPO_API_KEY'),
            'outputFormat': 'GTiff',
            'south': grid_bounds['geometry'].y[0],
            'north': grid_bounds['geometry'].y[1],
            'west': grid_bounds['geometry'].x[0],
            'east': grid_bounds['geometry'].x[1],
        }
        r = requests.get('https://portal.opentopography.org/API/usgsdem?datasetName=USGS10m', params=params)
        with open(f"{data_folder}{out_file}", "wb") as out_file:
            out_file.write(r.content)
    except Error:
        print(f"Failure to request data: {r.status_code}")
    return None


if __name__ == "__main__":
    download_dem("./data/kcm_realtime/processed/", [369903,37911,409618,87758], 32148, "usgs10m_dem.tif")