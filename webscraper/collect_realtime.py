import datetime
from pathlib import Path

import requests
import pandas as pd
import pytz


def collect_realtime_positions(data_folder, realtime_url, tz_str):
    data_folder.mkdir(parents=True, exist_ok=True)
    try:
        current_target_time = datetime.datetime.now(pytz.timezone(tz_str))
        current_epoch = round(current_target_time.timestamp())
        response = requests.get(realtime_url)
        with open(Path(data_folder, f"{current_target_time.strftime('%Y_%m_%d_%H')}_{current_epoch}.pb"), 'wb') as f:
            f.write(response.content)
    except:
        print(f"ERROR downloading {data_folder} realtime data")


if __name__ == "__main__":
    collect_realtime_positions(Path('open_bus_tools', 'webscraper', 'scraped_data', 'kcm_realtime', 'pbf'), 'https://s3.amazonaws.com/kcm-alerts-realtime-prod/vehiclepositions.pb', 'America/Los_Angeles')
    collect_realtime_positions(Path('open_bus_tools', 'webscraper', 'scraped_data', 'atb_realtime', 'pbf'), 'https://api.entur.io/realtime/v1/gtfs-rt/vehicle-positions?datasource=ATB', 'Europe/Oslo')
    cleaned_sources = pd.read_csv(Path('open_bus_tools', 'data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        collect_realtime_positions(Path('open_bus_tools', 'webscraper', 'scraped_data', 'other_feeds', f"{row['uuid']}_realtime", 'pbf'), row['realtime_url'], row['tz_str'])