import logging
from pathlib import Path
import pickle

import geopandas as gpd
import pandas as pd
import shapely
from srai.joiners import IntersectionJoiner
from srai.embedders import GTFS2VecEmbedder, Hex2VecEmbedder
from srai.loaders import GTFSLoader, OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer


def get_spatial_embeddings(spatial_folder, static_folder, grid_bounds, embedding_mode='pretrained'):
    """
    Downloads OpenStreetMap (OSM) pbf file covering the specified grid bounds. Saves the extracted OSM features to a pickle file.

    Args:
        data_folder (str): The path to the folder where the OSM features will be saved.
        grid_bounds (list): A list of four coordinates representing the bounding box of the grid in the format [min_x, min_y, max_x, max_y].
        out_file (str): The filename to save the extracted OSM features in.
        embedding_mode (str): The mode to use for the embeddings. Options are 'pretrained' and 'fit'. Defaults to 'pretrained'.
    """
    try:
        logger.debug(f"Creating regions for {spatial_folder}/{static_folder}")
        # Define area for embeddings, get H3 regions
        geo = shapely.Polygon((
            (grid_bounds[0], grid_bounds[1]),
            (grid_bounds[0], grid_bounds[3]),
            (grid_bounds[2], grid_bounds[3]),
            (grid_bounds[2], grid_bounds[1]),
            (grid_bounds[0], grid_bounds[1])
        ))
        area = gpd.GeoDataFrame({'region_id': [str(spatial_folder.parent)], 'geometry': [geo]}, crs='epsg:4326')
        area.set_index('region_id', inplace=True)
        regionalizer = H3Regionalizer(resolution=8)
        regions = regionalizer.transform(area)
        neighbourhood = H3Neighbourhood(regions_gdf=regions)
        # Save
        save_dir = spatial_folder / "spatial_embeddings"
        save_dir.mkdir(parents=True, exist_ok=True)
        area.to_pickle(save_dir / "area.pkl")
        regions.to_pickle(save_dir / "regions.pkl")
        with open(save_dir / 'neighbourhood.pkl', 'wb') as f:
            pickle.dump(neighbourhood, f)
    except:
        logger.debug(f"Failure to create regions for {spatial_folder}/{static_folder}")

    try:
        logger.debug(f"Embedding OSM for {spatial_folder}/{static_folder}")
        # Import OSM features and join to regions
        loader = OSMPbfLoader(download_source='protomaps', download_directory=spatial_folder / "spatial_embeddings")
        features = loader.load(area, HEX2VEC_FILTER)
        joiner = IntersectionJoiner()
        joint = joiner.transform(regions, features)
        # Either fit new embeddings or use pretrained from hex2vec
        if embedding_mode == 'pretrained':
            embedder_osm = Hex2VecEmbedder.load(Path("data", "pretrained_embeddings"))
        elif embedding_mode == 'fit':
            embedder_osm = Hex2VecEmbedder()
            embedder_osm.fit(regions, features, joint, neighbourhood, batch_size=128)
        embeddings_osm = embedder_osm.transform(regions, features, joint)
        # Save
        save_dir = spatial_folder / "spatial_embeddings"
        save_dir.mkdir(parents=True, exist_ok=True)
        embeddings_osm.to_pickle(save_dir / "embeddings_osm.pkl")
    except:
        logger.debug("Failure to embed OSM data")

    # try:
    #     logger.debug(f"Embedding GTFS for {spatial_folder}/{static_folder}")
    #     gtfs_folders = list(static_folder.glob('*'))
    #     for gtfs_folder in gtfs_folders:
    #         # Import GTFS features and join to regions
    #         loader = GTFSLoader()
    #         features = loader.load(gtfs_folder, skip_validation=True)
    #         features.index.name = 'feature_id'
    #         joiner = IntersectionJoiner()
    #         joint = joiner.transform(regions, features)
    #         embedder_gtfs = GTFS2VecEmbedder()
    #         embedder_gtfs.fit(regions, features, joint)
    #         embeddings_gtfs = embedder_gtfs.transform(regions, features, joint)
    #         # Save
    #         save_dir = spatial_folder / "spatial_embeddings"
    #         save_dir.mkdir(parents=True, exist_ok=True)
    #         embeddings_gtfs.to_pickle(save_dir / "embeddings_gtfs.pkl")
    # except:
    #     logger.debug("Failure to embed GTFS data")


if __name__ == "__main__":
    logger = logging.getLogger('get_spatial_embeddings')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Download KCM
    spatial_folder = Path('data', 'kcm_spatial')
    static_folder = Path('data', 'kcm_static')
    get_spatial_embeddings(spatial_folder, static_folder, [-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442])

    # Download ATB
    spatial_folder = Path('data', 'atb_spatial')
    static_folder = Path('data', 'atb_static')
    get_spatial_embeddings(spatial_folder, static_folder, [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395])

    # Download RUT
    spatial_folder = Path('data', 'rut_spatial')
    static_folder = Path('data', 'rut_static')
    get_spatial_embeddings(spatial_folder, static_folder, [10.588056382271377,59.809956950105395,10.875078411359919,59.95982169587328])

    # Download Others
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        spatial_folder = Path('data', 'other_feeds', f"{row['uuid']}_spatial")
        static_folder = Path('data', 'other_feeds', f"{row['uuid']}_static")
        get_spatial_embeddings(spatial_folder, static_folder, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']])