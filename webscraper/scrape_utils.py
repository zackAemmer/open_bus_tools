from datetime import datetime
from pathlib import Path
import uuid

import gtfs_realtime_pb2
import pandas as pd
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import pytz
import requests
import re
import timezonefinder



def get_time_info(tz=None):
    current_target_time = datetime.now(tz)
    date_str = current_target_time.strftime("%Y_%m_%d_%H")
    epoch = round(current_target_time.timestamp())
    return date_str, epoch, current_target_time


def datetime_to_epoch(time):
    # Go from time string in CET to epoch
    # format is '2022-03-16T16:14:58.12+01:00'
    yr = int(time[0:4])
    mo = int(time[5:7])
    day = int(time[8:10])
    hr = int(time[11:13])
    mn = int(time[14:16])
    sec = int(time[17:19])
    timezone = pytz.timezone('Europe/Oslo')
    ts = datetime(year=yr, month=mo, day=day, hour=hr, minute=mn, second=sec)
    ts = timezone.localize(ts)
    return ts.timestamp()


def xml_to_dict(element):
    # Recursively create a dictionary of XML field -> text value
    element_dict = {}
    for child in element:
        tag = re.split("}", child.tag)[1]
        if child.text != None:
            element_dict[tag] = child.text
        elif tag in element_dict.keys(): # In case multiple children with same tag exist in this element, turn into a list
            if type(element_dict[tag]) == list:
                element_dict[tag].append(xml_to_dict(child))
            else:
                first_elem = element_dict[tag]
                element_dict[tag] = []
                element_dict[tag].append(first_elem)
        else:
            element_dict[tag] = xml_to_dict(child)
    return element_dict


def clean_source_list():
    # Clean up the columns and join static + realtime urls
    more_gtfs_sources = pd.read_csv(Path('..', 'data', 'more_gtfs_sources.csv'))
    static_sources = more_gtfs_sources[more_gtfs_sources['data_type'] == "gtfs"]
    static_sources = static_sources[[
        'mdb_source_id',
        'location.country_code',
        'location.subdivision_name',
        'location.municipality',
        'provider',
        'urls.direct_download',
        'location.bounding_box.minimum_longitude',
        'location.bounding_box.maximum_longitude',
        'location.bounding_box.minimum_latitude',
        'location.bounding_box.maximum_latitude',
        'status'
    ]]
    realtime_sources = more_gtfs_sources[more_gtfs_sources['data_type'] == "gtfs-rt"]
    realtime_sources = realtime_sources[realtime_sources['entity_type']=='vp']
    realtime_sources = realtime_sources[[
        'static_reference',
        'location.country_code',
        'location.subdivision_name',
        'location.municipality',
        'provider',
        'urls.direct_download',
        'status'
    ]]
    joined_sources = pd.merge(static_sources, realtime_sources, left_on='mdb_source_id', right_on='static_reference')
    joined_sources = joined_sources[joined_sources['location.subdivision_name_x'] == joined_sources['location.subdivision_name_y']]
    joined_sources = joined_sources[joined_sources['provider_x'] == joined_sources['provider_y']]
    joined_sources = joined_sources[joined_sources['location.municipality_x'] == joined_sources['location.municipality_y']]
    joined_sources = joined_sources[joined_sources['status_x'] != "inactive"]
    joined_sources = joined_sources[joined_sources['status_y'] != "inactive"]
    joined_sources = joined_sources[[
        'location.country_code_x',
        'location.subdivision_name_x',
        'location.municipality_x',
        'provider_x',
        'urls.direct_download_x',
        'urls.direct_download_y',
        'location.bounding_box.minimum_longitude',
        'location.bounding_box.maximum_longitude',
        'location.bounding_box.minimum_latitude',
        'location.bounding_box.maximum_latitude'
    ]]
    joined_sources.columns = [
        'country_code',
        'subdivision_name',
        'municipality',
        'provider',
        'static_url',
        'realtime_url',
        'min_lon',
        'max_lon',
        'min_lat',
        'max_lat'
    ]
    # Find which realtime urls are not working
    broken_urls = []
    for i, url in enumerate(joined_sources['realtime_url']):
        try:
            response = requests.get(url)
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
        except:
            broken_urls.append(i)
    # Broken static feeds
    joined_sources = joined_sources[~joined_sources['provider'].isin(['Riverside Transit Agency','Transit Authority of Northern Kentucky (TANK)'])]
    # Save functional urls
    cleaned_sources = joined_sources.reset_index(drop=True).drop(broken_urls).reset_index(drop=True)
    cleaned_sources['uuid'] = [uuid.uuid4() for x in range(len(cleaned_sources))]
    cleaned_sources = cleaned_sources.dropna()
    cleaned_sources['tz_str'] = [timezonefinder.TimezoneFinder().timezone_at(lng=row['min_lon'], lat=row['min_lat']) for _, row in cleaned_sources.iterrows()]
    epsg_codes = []
    for _, row in cleaned_sources.iterrows():
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=row['min_lon'],
                south_lat_degree=row['min_lat'],
                east_lon_degree=row['max_lon'],
                north_lat_degree=row['max_lat'],
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        epsg_codes.append(utm_crs.to_epsg())
    cleaned_sources['epsg_code'] = epsg_codes
    cleaned_sources.to_csv(Path('..', 'data', 'other_feeds', 'cleaned_sources.csv'), index=False)