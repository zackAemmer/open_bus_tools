import datetime
from pathlib import Path
import zipfile

import requests
import pandas as pd
import pytz


def collect_static_schedules(data_folder, static_url, tz_str):
    data_folder.mkdir(exist_ok=True, parents=True)
    try:
        localized_date = datetime.datetime.now().astimezone(pytz.timezone(tz_str))
        localized_date_str = localized_date.strftime("%Y_%m_%d")
        response = requests.get(static_url)
        with open(Path(data_folder, f"{localized_date_str}.zip"), 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(Path(data_folder, f"{localized_date_str}.zip"), 'r') as zip_ref:
            destination_folder = Path(data_folder, f"{localized_date_str}")
            destination_folder.mkdir(exist_ok=True, parents=True)
            zip_ref.extractall(destination_folder)
    except:
        print(f"ERROR downloading {data_folder} static data")


if __name__ == "__main__":
    collect_static_schedules(Path('open_bus_tools', 'webscraper', 'scraped_data', 'kcm_static'), 'https://gtfs.sound.obaweb.org/prod/1_gtfs.zip', 'America/Los_Angeles')
    collect_static_schedules(Path('open_bus_tools', 'webscraper', 'scraped_data', 'atb_static'), 'https://storage.googleapis.com/marduk-production/outbound/gtfs/rb_atb-aggregated-gtfs.zip', 'Europe/Oslo')
    cleaned_sources = pd.read_csv(Path('open_bus_tools', 'data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        collect_static_schedules(Path('open_bus_tools', 'webscraper', 'scraped_data', 'other_feeds', f"{row['uuid']}_static"), row['static_url'], row['tz_str'])