import os
from pathlib import Path
import requests

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openbustools import spatial


def download_dem(endpoint, data_folder, grid_bounds, epsg, out_file):
    print(f"Getting DEM for {data_folder}{out_file} from {endpoint} at OpenTopography")
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
        r = requests.get(endpoint, params=params)
        with open(Path(data_folder, out_file), "wb") as out_file:
            out_file.write(r.content)
    except:
        print(f"Failure to request data: {r.status_code}")
    return None


if __name__ == "__main__":
    Path("./data/kcm_spatial").mkdir(parents=True, exist_ok=True)
    # download_dem("https://portal.opentopography.org/API/usgsdem?datasetName=USGS10m", "./data/kcm_spatial/", [369903,37911,409618,87758], 32148, "usgs10m_dem.tif")
    # spatial.reproject_raster("./data/kcm_spatial/usgs10m_dem.tif", "./data/kcm_spatial/usgs10m_dem_32148.tif", 32148)
    download_dem("https://portal.opentopography.org/API/globaldem?demtype=EU_DTM", "./data/atb_spatial/", [550869,7012847,579944,7039521], 32632, "eudtm30m_dem.tif")
    spatial.reproject_raster("./data/atb_spatial/eudtm30m_dem.tif", "./data/atb_spatial/eudtm30m_dem_32632.tif", 32632)
    download_dem("https://portal.opentopography.org/API/globaldem?demtype=EU_DTM", "./data/rut_spatial/", [589080,6631314,604705,6648420], 32632, "eudtm30m_dem.tif")
    spatial.reproject_raster("./data/rut_spatial/eudtm30m_dem.tif", "./data/rut_spatial/eudtm30m_dem_32632.tif", 32632)