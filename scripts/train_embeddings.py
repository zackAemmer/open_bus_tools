import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from srai.joiners import IntersectionJoiner
from srai.embedders import GTFS2VecEmbedder, Hex2VecEmbedder
from srai.loaders import GTFSLoader, OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER

from openbustools import spatial


def fit_pretrained_gtfs_embeddings(embeddings_folder, static_folder, model_folder):
    logger.debug(f"Embedding GTFS for {embeddings_folder}")
    loader = GTFSLoader()
    joiner = IntersectionJoiner()
    embedder_gtfs = GTFS2VecEmbedder.load(model_folder)
    # Load precalculated regions
    area, regions, neighbourhood = spatial.load_regions(embeddings_folder)
    # Calculate embeddings for each separate GTFS feed date
    gtfs_file = [x for x in list(static_folder.glob('*.zip'))][1]
    # Extract GTFS features and join to regions
    features = loader.load(gtfs_file=gtfs_file, skip_validation=True)
    # features = add_missing_embedding_columns(features)
    joint = joiner.transform(regions, features)
    # The GTFS embedder breaks when certain hours have either no directions or trips
    d_varnames = [f"directions_at_{i}" for i in range(24)]
    for var in d_varnames:
        if var not in features.columns:
            features[var] = np.nan
    t_varnames = [f"trips_at_{i}" for i in range(24)]
    for var in t_varnames:
        if var not in features.columns:
            features[var] = 0.0
    t_varnames.append("geometry")
    t_varnames.extend(d_varnames)
    features = features[t_varnames]
    # Use pretrained embedder to transform features to embeddings
    embeddings_gtfs = embedder_gtfs.transform(regions, features, joint)
    try:
        assert(embeddings_gtfs.isna().sum().sum()==0) # Check that all regions have embeddings
    except:
        logger.error(f"Error embedding GTFS: {embeddings_gtfs.isna().sum().sum()} missing values")
        return
    embeddings_gtfs.to_pickle(embeddings_folder / 'embeddings_gtfs.pkl')


def fit_pretrained_osm_embeddings(embeddings_folder, model_folder):
    logger.debug(f"Embedding OSM for {embeddings_folder}")
    loader = OSMPbfLoader(download_source='geofabrik', download_directory=Path('data','osm_dl'))
    joiner = IntersectionJoiner()
    embedder_osm = Hex2VecEmbedder.load(model_folder)
    # Load precalculated regions
    area, regions, neighbourhood = spatial.load_regions(embeddings_folder)
    # Download OSM pbf file if needed, extract and join regions
    features = loader.load(area=area, tags=HEX2VEC_FILTER)
    joint = joiner.transform(regions, features)
    # Use pretrained embedder to transform features to embeddings
    embeddings_osm = embedder_osm.transform(regions, features, joint)
    assert(embeddings_osm.isna().sum().sum()==0) # Check that all regions have embeddings
    # Save
    embeddings_osm.to_pickle(embeddings_folder / "embeddings_osm.pkl")


def train_combined_gtfs_embedder(embeddings_folders, save_dir):
    logger.debug(f"Creating combined GTFS embedder")
    save_dir.mkdir(parents=True, exist_ok=True)
    all_regions = []
    all_features = []
    loader = GTFSLoader()
    joiner = IntersectionJoiner()
    embedder_gtfs = GTFS2VecEmbedder(embedding_size=64)
    # Load all GTFS feeds and regions
    for i, folder in enumerate(embeddings_folders):
        logger.debug(f"Loading area, features from {folder}")
        try:
            # Get the previously calculated hex regions for each feed
            embeddings_dir = folder / "spatial_embeddings"
            _, regions, _ = spatial.load_regions(embeddings_dir)
            # Get GTFS information for each feed
            static_dir = folder.parent / f"{folder.name.split('_')[0]}_static"
            features = loader.load([x for x in static_dir.glob('*.zip')][0], skip_validation=True)
            # If both load successfully, add to training feeds
            all_regions.append(regions)
            all_features.append(features)
        except Exception as e:
            logger.error(f"Error loading {folder}: {e}")
    logger.debug(f"Combining all areas, regions, features and fitting GTFS embedder")
    all_regions = pd.concat(all_regions)
    all_features = pd.concat(all_features)
    joint = joiner.transform(all_regions, all_features)
    # Train embedder on all combined GTFS feeds
    embedder_gtfs.fit(all_regions, all_features, joint)
    embedder_gtfs.save(save_dir)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('get_spatial_embeddings')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    cleaned_sources = pd.read_csv(Path('data','cleaned_sources.csv'))

    # # Create hex regions for embedding each feed
    # _, _, _ = spatial.create_regions([-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], Path('data','kcm_spatial','spatial_embeddings'))
    # _, _, _ = spatial.create_regions([10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], Path('data','atb_spatial','spatial_embeddings'))
    # for i, row in cleaned_sources.iterrows():
    #     spatial.create_regions([row['min_lon'],row['min_lat'],row['max_lon'],row['max_lat']], Path('data','other_feeds',f"{row['uuid']}_spatial","spatial_embeddings"))

    # # Train a GTFS embedder on all feeds
    # embeddings_folders = [Path('data','kcm_spatial'), Path('data','atb_spatial')]
    # embeddings_folders.extend([Path('data','other_feeds',f"{uuid}_spatial") for uuid in cleaned_sources['uuid']])
    # train_combined_gtfs_embedder(embeddings_folders, Path('data','pretrained_embeddings','gtfs2vec'))

    # # Fit OSM embeddings
    # fit_pretrained_osm_embeddings(Path('data','kcm_spatial','spatial_embeddings'), Path('data','pretrained_embeddings','hex2vec','10','hex2vec_poland_10_50k'))
    # fit_pretrained_osm_embeddings(Path('data','atb_spatial','spatial_embeddings'), Path('data','pretrained_embeddings','hex2vec','10','hex2vec_poland_10_50k'))
    # for i, row in cleaned_sources.iterrows():
    #     fit_pretrained_osm_embeddings(Path('data','other_feeds',f"{row['uuid']}_spatial",'spatial_embeddings'), Path('data','pretrained_embeddings','hex2vec','10','hex2vec_poland_10_50k'))

    # Fit GTFS embeddings
    # fit_pretrained_gtfs_embeddings(Path('data','kcm_spatial','spatial_embeddings'), Path('data','kcm_static'), Path('data','pretrained_embeddings','gtfs2vec'))
    # fit_pretrained_gtfs_embeddings(Path('data','atb_spatial','spatial_embeddings'), Path('data','atb_static'), Path('data','pretrained_embeddings','gtfs2vec'))
    for i, row in cleaned_sources.iterrows():
        fit_pretrained_gtfs_embeddings(Path('ExtremeSSD','data','other_feeds',f"{row['uuid']}_spatial",'spatial_embeddings'), Path('ExtremeSSD','data','other_feeds',f"{row['uuid']}_static"), Path('ExtremeSSD','data','pretrained_embeddings','gtfs2vec'))