import pickle
import pytz
import requests
from xml.etree import ElementTree

from dotenv import load_dotenv
import pandas as pd

import scrape_utils, gtfs_realtime_pb2


# with open(f"./data/trip-updates.pbf", 'rb') as f:
#     veh_pos = gtfs_realtime_pb2.FeedMessage()
#     veh_pos.ParseFromString(f.read())

if __name__ == "__main__":
    load_dotenv()
