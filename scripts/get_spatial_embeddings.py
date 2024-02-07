import logging
from pathlib import Path
import shutil

import lightning.pytorch as pl
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
    # # Remove existing embeddings if desired
    # dir_to_wipe = Path(spatial_folder, "spatial_embeddings")
    # if dir_to_wipe.exists():
    #     shutil.rmtree(dir_to_wipe)
    # [x.unlink() for x in spatial_folder.glob('*.pkl')]

    # Create embeddings directory if doesn't exist
    embeddings_dir = Path(spatial_folder, "spatial_embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Check if downloaded properly
    osm_dir = list(embeddings_dir.glob('Geofabrik*'))
    embeddings = list(embeddings_dir.glob('embeddings_osm.pkl'))
    print(len(osm_dir), len(embeddings))
    if len(osm_dir) == 0 or len(embeddings) == 0:
        print("Download OSM")

    if not create_new_regions:
        try:
            logger.debug(f"Loading regions for {spatial_folder} and {static_folder}")
            area, regions, neighbourhood = spatial.load_regions(embeddings_dir)
        except:
            logger.debug(f"Existing regions not found, creating regions for {spatial_folder}/{static_folder}")
            area, regions, neighbourhood = spatial.create_regions(grid_bounds, embeddings_dir)
    else:
        logger.debug(f"Creating regions for {spatial_folder}/{static_folder}")
        area, regions, neighbourhood = spatial.create_regions(grid_bounds, embeddings_dir)

    # try:
    #     logger.debug(f"Embedding OSM for {spatial_folder} and {static_folder}")
    #     # Search for existing OSM pbf files
    #     if len(list(embeddings_dir.glob('*.osm.pbf'))) > 0:
    #         loader = OSMPbfLoader(pbf_file=list(embeddings_dir.glob('*.osm.pbf'))[0])
    #     else:
    #         loader = OSMPbfLoader(download_source='geofabrik', download_directory=embeddings_dir)
    #     # Download OSM pbf file if needed, extract and join regions
    #     features = loader.load(area, HEX2VEC_FILTER)
    #     joiner = IntersectionJoiner()
    #     joint = joiner.transform(regions, features)
    #     # Either fit new embeddings or use pretrained embeddings from hex2vec
    #     if embedding_mode == 'pretrained':
    #         embedder_osm = Hex2VecEmbedder.load(Path("data", "pretrained_embeddings"))
    #     elif embedding_mode == 'fit':
    #         embedder_osm = Hex2VecEmbedder()
    #         embedder_osm.fit(regions, features, joint, neighbourhood, batch_size=128)
    #     embeddings_osm = embedder_osm.transform(regions, features, joint)
    #     # Save
    #     embeddings_osm.to_pickle(embeddings_dir / "embeddings_osm.pkl")
    # except Exception as e:
    #     logger.debug(e)

    # try:
    #     logger.debug(f"Embedding GTFS for {spatial_folder}/{static_folder}")
    #     gtfs_folders = list(static_folder.glob('*'))
    #     # Need embeddings for each separate GTFS feed date
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
    #         embeddings_gtfs.to_pickle(embeddings_dir / f"embeddings_gtfs_{gtfs_folder.name}.pkl")
    # except Exception as e:
    #     logger.debug(e)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('get_spatial_embeddings')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Download KCM
    get_spatial_embeddings(Path('data','kcm_spatial'), Path('data','kcm_static'), [-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], create_new_regions=False)

    # Download ATB
    get_spatial_embeddings(Path('data','atb_spatial'), Path('data','atb_static'), [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], create_new_regions=False)

    # Download Others
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        spatial_folder = Path('data','other_feeds',f"{row['uuid']}_spatial")
        static_folder = Path('data','other_feeds',f"{row['uuid']}_static")
        get_spatial_embeddings(spatial_folder, static_folder, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], create_new_regions=False)