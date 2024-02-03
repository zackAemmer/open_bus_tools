import logging
from pathlib import Path

import pandas as pd
from srai.joiners import IntersectionJoiner
from srai.embedders import GTFS2VecEmbedder, Hex2VecEmbedder
from srai.loaders import GTFSLoader, OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER

from openbustools import spatial


def get_spatial_embeddings(spatial_folder, static_folder, grid_bounds, embedding_mode='pretrained', create_new_regions=False):
    """
    Downloads OpenStreetMap (OSM) pbf file covering the specified grid bounds. Saves the extracted OSM features to a pickle file.

    Args:
        spatial_folder (Path): The folder to save the spatial embeddings to.
        static_folder (Path): The folder containing the static schedule data.
        grid_bounds (list): A list of four coordinates representing the bounding box of the grid in the format [min_x, min_y, max_x, max_y].
        embedding_mode (str): The mode to use for the embeddings. Options are 'pretrained' and 'fit'. Defaults to 'pretrained'.
        create_new_regions (bool): Whether to create new regions or use existing ones if found. Defaults to False.
    """
    if not create_new_regions:
        try:
            logger.debug(f"Loading regions for {spatial_folder} and {static_folder}")
            area, regions, neighbourhood = spatial.load_regions(spatial_folder)
        except:
            logger.debug(f"Existing regions not found, creating regions for {spatial_folder}/{static_folder}")
            area, regions, neighbourhood = spatial.create_regions(grid_bounds, spatial_folder)
    else:
        logger.debug(f"Creating regions for {spatial_folder}/{static_folder}")
        area, regions, neighbourhood = spatial.create_regions(grid_bounds, spatial_folder)

    try:
        logger.debug(f"Embedding OSM for {spatial_folder} and {static_folder}")
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
    except Exception as e:
        logger.debug(e)

    # try:
    #     logger.debug(f"Embedding GTFS for {spatial_folder}/{static_folder}")
    #     gtfs_folders = list(static_folder.glob('*'))
    #     for gtfs_folder in gtfs_folders:
    #         # Import GTFS features and join to regions
    #         loader = GTFSLoader()
    #         features = loader.load(gtfs_folder, skip_validation=True)
    #         joiner = IntersectionJoiner()
    #         joint = joiner.transform(regions, features)
    #         embedder_gtfs = GTFS2VecEmbedder()
    #         embedder_gtfs.fit(regions, features, joint)
    #         embeddings_gtfs = embedder_gtfs.transform(regions, features, joint)
    #         # Save
    #         save_dir = spatial_folder / "spatial_embeddings"
    #         save_dir.mkdir(parents=True, exist_ok=True)
    #         embeddings_gtfs.to_pickle(save_dir / "embeddings_gtfs.pkl")
    # except Exception as e:
    #     logger.debug(e)


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
    get_spatial_embeddings(spatial_folder, static_folder, [-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], create_new_regions=True)

    # Download ATB
    spatial_folder = Path('data', 'atb_spatial')
    static_folder = Path('data', 'atb_static')
    get_spatial_embeddings(spatial_folder, static_folder, [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], create_new_regions=True)

    # Download RUT
    spatial_folder = Path('data', 'rut_spatial')
    static_folder = Path('data', 'rut_static')
    get_spatial_embeddings(spatial_folder, static_folder, [10.588056382271377,59.809956950105395,10.875078411359919,59.95982169587328], create_new_regions=True)

    # Download Others
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        spatial_folder = Path('data', 'other_feeds', f"{row['uuid']}_spatial")
        static_folder = Path('data', 'other_feeds', f"{row['uuid']}_static")
        get_spatial_embeddings(spatial_folder, static_folder, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], create_new_regions=True)