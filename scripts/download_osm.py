import logging
from pathlib import Path
import pickle

import geopandas as gpd
import pandas as pd
import shapely
from srai.loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER


def download_osm(data_folder, grid_bounds, out_file):
    """
    Downloads OpenStreetMap (OSM) pbf file covering the specified grid bounds. Saves the extracted OSM features to a pickle file.

    Args:
        data_folder (str): The path to the folder where the OSM features will be saved.
        grid_bounds (list): A list of four coordinates representing the bounding box of the grid in the format [min_x, min_y, max_x, max_y].
        out_file (str): The filename to save the extracted OSM features in.
    """
    logger.debug(f"Downloading/Extracting OSM features for {data_folder} to {out_file}")
    # Define area for embeddings
    geo = shapely.Polygon((
        (grid_bounds[0], grid_bounds[1]),
        (grid_bounds[0], grid_bounds[3]),
        (grid_bounds[2], grid_bounds[3]),
        (grid_bounds[2], grid_bounds[1]),
        (grid_bounds[0], grid_bounds[1])
    ))
    area = gpd.GeoDataFrame({'region_id': [str(data_folder)], 'geometry': [geo]}, crs='epsg:4326')
    area.set_index('region_id', inplace=True)
    # Import OSM features and join to regions
    loader = OSMPbfLoader(download_source='protomaps', download_directory=data_folder)
    features = loader.load(area, HEX2VEC_FILTER)
    features.to_pickle(Path(data_folder, out_file))
    logger.debug(f"Found {len(features)} features")


if __name__ == "__main__":
    logger = logging.getLogger('download_osm')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Download KCM
    data_folder = Path('data', 'kcm_spatial')
    data_folder.mkdir(parents=True, exist_ok=True)
    download_osm(data_folder, [-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], 'osm_features_hex2vec.pkl')

    # Download ATB
    data_folder = Path('data', 'atb_spatial')
    data_folder.mkdir(parents=True, exist_ok=True)
    download_osm(data_folder, [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], 'osm_features_hex2vec.pkl')

    # Download RUT
    data_folder = Path('data', 'rut_spatial')
    data_folder.mkdir(parents=True, exist_ok=True)
    download_osm(data_folder, [10.588056382271377,59.809956950105395,10.875078411359919,59.95982169587328], 'osm_features_hex2vec.pkl')

    # Download Others
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        data_folder = Path('data', 'other_feeds', f"{row['uuid']}_spatial")
        data_folder.mkdir(parents=True, exist_ok=True)
        download_osm(data_folder, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], 'osm_features_hex2vec.pkl')