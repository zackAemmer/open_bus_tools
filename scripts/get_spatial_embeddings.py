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


# # Remove existing embeddings if desired
# dir_to_wipe = Path(spatial_folder, "spatial_embeddings")
# if dir_to_wipe.exists():
#     shutil.rmtree(dir_to_wipe)
# [x.unlink() for x in spatial_folder.glob('*.pkl')]

# # Check if downloaded properly
# osm_dir = list(embeddings_dir.glob('Geofabrik*'))
# embeddings = list(embeddings_dir.glob('embeddings_osm.pkl'))
# print(len(osm_dir), len(embeddings))
# if len(osm_dir) == 0 or len(embeddings) == 0:
#     print(embeddings_dir.parent)
#     print("Download OSM")

# # My temporary fix for GTFSLoader issues where the first wednesday does not have any trips
# # Can be caused by service alerts or calendar starting way prior to actual trips
# # Only use start date range tied to service ids; service alerts go back before actual trips start
# dates = []
# if feed.calendar is not None and not feed.calendar.empty:
#     if "start_date" in feed.calendar.columns:
#         dates.append(feed.calendar["start_date"].min())
#     if "end_date" in feed.calendar.columns:
#         dates.append(feed.calendar["end_date"].max())
#     # Get first wednesday that falls in service range, put upper limit
#     for n in range(1, 365*2):
#         date = feed.get_week(n)[2]
#         if date > min(dates):
#             break
# # Fall back to first wednesday
# else:
#     date = feed.get_first_week()[2]
# # Start from first reasonable date, test each 3 days until find a date with trips
# import datetime
# try_date = datetime.datetime.strptime(date, "%Y%m%d")
# for i in range(90):
#     ts = feed.compute_stop_time_series([try_date.strftime("%Y%m%d")], freq=self.time_resolution)
#     if ts.sum().sum()!=0:
#         break
#     try_date = try_date + datetime.timedelta(days=3)


def create_combined_gtfs_embedder(cleaned_sources, kcm_embeddings_folder, atb_embeddings_folder):
    logger.debug(f"Creating combined GTFS embedder")
    all_areas = []
    all_regions = []
    all_features = []
    loader = GTFSLoader()
    joiner = IntersectionJoiner()
    embedder_gtfs = GTFS2VecEmbedder(embedding_size=16)
    # Train embedder on all combined GTFS feeds
    for i, folder in enumerate([kcm_embeddings_folder, atb_embeddings_folder]):
        logger.debug(f"Loading area, features from {folder}")
        static_dir = folder.parent / f"{folder.name.split('_')[0]}_static"
        embeddings_dir = Path(folder, "spatial_embeddings")
        area, regions, neighbourhood = spatial.load_regions(embeddings_dir)
        feed_dates = list(set(static_dir.glob('*')) - set(static_dir.glob('*.zip')))
        feed_date = feed_dates[0]
        logger.debug(f"Using feed from {feed_date}")
        features = loader.load(feed_date, skip_validation=True)
        all_areas.append(area)
        all_regions.append(regions)
        all_features.append(features)
    for i, row in cleaned_sources.iterrows():
        logger.debug(f"Loading area, features from {row['uuid']}")
        static_dir = Path('data','other_feeds',f"{row['uuid']}_static")
        embeddings_dir = Path('data','other_feeds',f"{row['uuid']}_spatial", "spatial_embeddings")
        area, regions, neighbourhood = spatial.load_regions(embeddings_dir)
        feed_dates = list(set(static_dir.glob('*')) - set(static_dir.glob('*.zip')))
        feed_date = feed_dates[0]
        logger.debug(f"Using feed from {feed_date}")
        features = loader.load(feed_date, skip_validation=True)
        all_areas.append(area)
        all_regions.append(regions)
        all_features.append(features)
    logger.debug(f"Combining all areas, regions, features and fitting GTFS embedder")
    all_areas = pd.concat(all_areas)
    all_regions = pd.concat(all_regions)
    all_features = pd.concat(all_features)
    # Fit embedder to all GTFS feeds and save
    joint = joiner.transform(regions, features)
    embedder_gtfs.fit(regions, features, joint)
    embedder_gtfs.save(Path("data", "pretrained_embeddings", "gtfs2vec"))


def calculate_gtfs_embeddings(spatial_folder, static_folder):
    logger.debug(f"Embedding GTFS for {static_folder}")
    embeddings_dir = Path(spatial_folder, "spatial_embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    loader = GTFSLoader()
    joiner = IntersectionJoiner()
    embedder_gtfs = GTFS2VecEmbedder.load(Path("data", "pretrained_embeddings", "gtfs2vec"))
    # Load precalculated regions
    area, regions, neighbourhood = spatial.load_regions(embeddings_dir)
    # Calculate embeddings for each separate GTFS feed date
    gtfs_folders = [x for x in list(static_folder.glob('*')) if x.is_dir()]
    for gtfs_file in gtfs_folders:
        logger.debug(f"Embedding GTFS for {gtfs_file}")
        # Extract GTFS features and join to regions
        features = loader.load(gtfs_file=gtfs_file, skip_validation=True)
        joint = joiner.transform(regions, features)
        # Use pretrained embedder to transform features to embeddings
        embedder_gtfs.fit(regions, features, joint)
        embeddings_gtfs = embedder_gtfs.transform(regions, features, joint)
        assert(embeddings_gtfs.isna().sum().sum()==0) # Check that all regions have embeddings
        # Save
        embeddings_gtfs.to_pickle(embeddings_dir / f"embeddings_gtfs_{gtfs_file.name}.pkl")


def calculate_osm_embeddings(spatial_folder):
    logger.debug(f"Embedding OSM for {spatial_folder}")
    embeddings_dir = Path(spatial_folder, "spatial_embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    loader = OSMPbfLoader(download_source='geofabrik', download_directory=embeddings_dir)
    joiner = IntersectionJoiner()
    embedder_osm = Hex2VecEmbedder.load(Path("data", "pretrained_embeddings"))
    # Load precalculated regions
    area, regions, neighbourhood = spatial.load_regions(embeddings_dir)
    # Download OSM pbf file if needed, extract and join regions
    features = loader.load(area=area, tags=HEX2VEC_FILTER)
    joint = joiner.transform(regions, features)
    # Use pretrained embedder to transform features to embeddings
    embeddings_osm = embedder_osm.transform(regions, features, joint)
    assert(embeddings_osm.isna().sum().sum()==0) # Check that all regions have embeddings
    # Save
    embeddings_osm.to_pickle(embeddings_dir / "embeddings_osm.pkl")


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    logger = logging.getLogger('get_spatial_embeddings')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))

    # # Create regions
    # _, _, _ = spatial.create_regions([-122.55451384931364,47.327892566537194,-122.0395248374609,47.78294919355442], Path('data','kcm_spatial'))
    # _, _, _ = spatial.create_regions([10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], Path('data','atb_spatial'))
    # for i, row in cleaned_sources.iterrows():
    #     spatial.create_regions([row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], Path('data','other_feeds',f"{row['uuid']}_spatial"))

    # Create one combined GTFS embedder
    create_combined_gtfs_embedder(cleaned_sources, Path('data','kcm_spatial'), Path('data','atb_spatial'))

    # Get OSM embeddings
    calculate_osm_embeddings(Path('data','kcm_spatial'))
    calculate_osm_embeddings(Path('data','atb_spatial'))
    for i, row in cleaned_sources.iterrows():
        calculate_osm_embeddings(Path('data','other_feeds',f"{row['uuid']}_spatial"))

    # Get GTFS embeddings
    calculate_gtfs_embeddings(Path('data', 'kcm_spatial'), Path('data', 'kcm_static'))
    calculate_gtfs_embeddings(Path('data', 'atb_spatial'), Path('data', 'atb_static'))
    for i, row in cleaned_sources.iterrows():
        calculate_gtfs_embeddings(Path('data', 'other_feeds', f"{row['uuid']}_spatial"), Path('data','other_feeds',f"{row['uuid']}_static"))