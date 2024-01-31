from dotenv import load_dotenv
load_dotenv()
import geopandas as gpd
import pandas as pd
from pathlib import Path
from rasterio.plot import show
import shapely

from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer
from srai.joiners import IntersectionJoiner
from srai.embedders import Hex2VecEmbedder, GTFS2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader, GTFSLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.plotting import plot_regions, plot_numeric_data

import logging
from pathlib import Path
import pickle

import lightning.pytorch as lp
import numpy as np
import pandas as pd

from openbustools import spatial, standardfeeds, trackcleaning
from openbustools.traveltime import grid


def process_data(**kwargs):
    logger.debug(f"PROCESSING: {kwargs['network_name']}")
    for day in kwargs['dates']:
        logger.debug(f"DAY: {day}")

        # Define area for embeddings
        geo = shapely.Polygon((
            (row['min_lon'],row['min_lat']),
            (row['min_lon'],row['max_lat']),
            (row['max_lon'],row['max_lat']),
            (row['max_lon'],row['min_lat']),
            (row['min_lon'],row['min_lat'])
        ))
        area = gpd.GeoDataFrame({'uuid': [row['uuid']], 'region_id': [f"{row['municipality']}, {row['subdivision_name']}, {row['country_code']}"], 'geometry': [geo]}, crs='epsg:4326')
        area.set_index('region_id', inplace=True)
        regionalizer = H3Regionalizer(resolution=8)
        joiner = IntersectionJoiner()
        regions = regionalizer.transform(area)
        neighbourhood = H3Neighbourhood(regions_gdf=regions)

        # Load GTFS features and join to regions
        loader = GTFSLoader()
        gtfs_file = list(Path('data', 'other_feeds', f"{row['uuid']}_static").glob('*.zip'))[0]
        features = loader.load(gtfs_file, skip_validation=False)
        joint = joiner.transform(regions, features)
        # Need to name the index in features; possible bug in GTFSLoader?
        features.index.name = 'feature_id'
        # Also had to change code in embedder: agg_dict[column] = lambda x: len(reduce(set.union, (val for val in x if not pd.isna(val)), set()))
        # Fit and transform GTFS features to regions with GTFS2Vec
        embedder = GTFS2VecEmbedder(hidden_size=2, embedding_size=4)
        embedder.fit(regions, features, joint)
        embeddings_gtfs = embedder.transform(regions, features, joint)
        # Plot the embeddings
        folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
        plot_numeric_data(regions, 0, embeddings_gtfs, map=folium_map)
        folium_map
        # Save the map as an HTML file
        folium_map.save("map_gtfs.html")

        # Import OSM features and join to regions
        loader = OSMPbfLoader()
        features = loader.load(area, HEX2VEC_FILTER)
        joint = joiner.transform(regions, features)
        # Embed OSM features to regions with Hex2Vec
        embedder_osm = Hex2VecEmbedder([15, 10, 3])
        embedder_osm.fit(regions, features, joint, neighbourhood, batch_size=128)
        embeddings_osm = embedder_osm.transform(regions, features, joint)
        # Plot the embeddings
        folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
        plot_numeric_data(regions, 0, embeddings_osm, map=folium_map)
        folium_map
        # Save the map as an HTML file
        folium_map.save("map_osm.html")

if __name__=="__main__":
    lp.seed_everything(42, workers=True)
    logger = logging.getLogger('process_data')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        if i != 2:
            continue
        if standardfeeds.validate_realtime_data(row):
            logger.debug(f"{row['provider']}")
            process_data(
                network_name=row['uuid'],
                dates=standardfeeds.get_date_list("2024_01_03", 5),
                static_folder=Path('data', 'other_feeds', f"{row['uuid']}_static"),
                realtime_folder=Path('data', 'other_feeds', f"{row['uuid']}_realtime"),
                dem_file=[x for x in Path('data', 'other_feeds', f"{row['uuid']}_spatial").glob(f"*_{row['epsg_code']}.tif")][0],
                timezone=row['tz_str'],
                epsg=row['epsg_code'],
                grid_bounds=[row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']],
                coord_ref_center=[np.mean([row['min_lon'], row['max_lon']]), np.mean([row['min_lat'], row['max_lat']])],
                given_names=['trip_id','file','locationtime','lat','lon','vehicle_id'],
            )