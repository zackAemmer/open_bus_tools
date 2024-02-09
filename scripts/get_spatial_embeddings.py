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

    # # Debug specific folder
    # spatial_folder = spatial_folder.parent / "6885d33d-2f4e-4ac7-84a7-2e4747c723f0_spatial"
    # static_folder = static_folder.parent / "6885d33d-2f4e-4ac7-84a7-2e4747c723f0_static"

    # Create or load hex regions and embeddings directory
    embeddings_dir = Path(spatial_folder, "spatial_embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
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

    # Embed OSM data
    logger.debug(f"Embedding OSM for {spatial_folder} and {static_folder}")
    loader = OSMPbfLoader(download_source='geofabrik', download_directory=embeddings_dir)
    # Download OSM pbf file if needed, extract and join regions
    features = loader.load(area, HEX2VEC_FILTER)
    joiner = IntersectionJoiner()
    joint = joiner.transform(regions, features)
    # Either fit new embeddings or use pretrained embeddings from hex2vec
    if embedding_mode == 'pretrained':
        embedder_osm = Hex2VecEmbedder.load(Path("data", "pretrained_embeddings"))
    elif embedding_mode == 'fit':
        embedder_osm = Hex2VecEmbedder()
        embedder_osm.fit(regions, features, joint, neighbourhood, batch_size=128)
    embeddings_osm = embedder_osm.transform(regions, features, joint)
    assert(embeddings_osm.isna().sum().sum()==0) # Check that all regions have embeddings
    # Save
    embeddings_osm.to_pickle(embeddings_dir / "embeddings_osm.pkl")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,1)
    regions.plot(ax=axes, color='red')
    features.plot(ax=axes, color='blue')
    fig.savefig("gtfs_join.png")

    # Embed GTFS feeds
    logger.debug(f"Embedding GTFS for {spatial_folder}/{static_folder}")
    gtfs_folders = static_folder.glob('*')
    gtfs_folders = [x for x in gtfs_folders if x.is_dir()]

    # Need embeddings for each separate GTFS feed date
    for gtfs_folder in gtfs_folders:
        # Import GTFS features and join to regions
        loader = GTFSLoader()
        logger.debug(f"Trying folder: {gtfs_folder}")
        features = loader.load(gtfs_folder, skip_validation=True, area=area)
        joiner = IntersectionJoiner()
        joint = joiner.transform(regions, features)
        # Fit embedder to GTFS feed
        embedder_gtfs = GTFS2VecEmbedder()
        embedder_gtfs.fit(regions, features, joint)
        embeddings_gtfs = embedder_gtfs.transform(regions, features, joint)
        assert(embeddings_gtfs.isna().sum().sum()==0) # Check that all regions have embeddings
        # Save
        embeddings_gtfs.to_pickle(embeddings_dir / f"embeddings_gtfs_{gtfs_folder.name}.pkl")


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
    get_spatial_embeddings(Path('data','atb_spatial'), Path('data','nwy_static'), [10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395], create_new_regions=False)

    # Download Others
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        spatial_folder = Path('data','other_feeds',f"{row['uuid']}_spatial")
        static_folder = Path('data','other_feeds',f"{row['uuid']}_static")
        get_spatial_embeddings(spatial_folder, static_folder, [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']], create_new_regions=False)