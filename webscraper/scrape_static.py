import datetime
from pathlib import Path
import zipfile

import requests
import pandas as pd
import pytz


if __name__ == "__main__":
    # Load cleaned list of transit data providers
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    # For each transit feed, query static data, unzip and save to folder
    for index, row in cleaned_sources.iterrows():
        provider_folder = Path('data', 'other_feeds', f"{row['uuid']}_static")
        provider_folder.mkdir(exist_ok=True, parents=True)
        localized_date = datetime.datetime.now().astimezone(pytz.timezone(row['tz_str']))
        localized_date_str = localized_date.strftime("%Y_%m_%d")

        # Download and extract recent zip feed
        try:
            response = requests.get(row['static_url'])
            with open(Path(provider_folder, f"{localized_date_str}.zip"), 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(Path(provider_folder, f"{localized_date_str}.zip"), 'r') as zip_ref:
                destination_folder = Path(provider_folder, f"{localized_date_str}")
                destination_folder.mkdir(exist_ok=True, parents=True)
                zip_ref.extractall(destination_folder)
        except:
            print(f"ERROR downloading {row['provider']} static data")
            continue
        # # Extract all static zip feeds
        # try:
        #     static_zips = Path(provider_folder).glob("*.zip")
        #     for sz in static_zips:
        #         localized_date_str = sz.stem
        #         with zipfile.ZipFile(Path(provider_folder, f"{localized_date_str}.zip"), 'r') as zip_ref:
        #             destination_folder = Path(provider_folder, f"{localized_date_str}")
        #             destination_folder.mkdir(exist_ok=True, parents=True)
        #             zip_ref.extractall(destination_folder)
        # except Exception as e:
        #     print(f"ERROR unzipping {row['provider']} static data {e}")
        #     continue

    # Repeat for KCM
    provider_folder = Path('data', 'kcm_static')
    provider_folder.mkdir(exist_ok=True, parents=True)
    localized_date = datetime.datetime.now().astimezone(pytz.timezone("America/Los_Angeles"))
    localized_date_str = localized_date.strftime("%Y_%m_%d")
    try:
        response = requests.get("https://gtfs.sound.obaweb.org/prod/1_gtfs.zip")
        with open(Path(provider_folder, f"{localized_date_str}.zip"), 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(Path(provider_folder, f"{localized_date_str}.zip"), 'r') as zip_ref:
            destination_folder = Path(provider_folder, f"{localized_date_str}")
            destination_folder.mkdir(exist_ok=True, parents=True)
            zip_ref.extractall(destination_folder)
    except:
        print(f"ERROR downloading KCM static data")

    # Repeat for NWY
    provider_folder = Path('data', 'nwy_static')
    provider_folder.mkdir(exist_ok=True, parents=True)
    localized_date = datetime.datetime.now().astimezone(pytz.timezone("Europe/Oslo"))
    localized_date_str = localized_date.strftime("%Y_%m_%d")
    try:
        response = requests.get("https://storage.googleapis.com/marduk-production/outbound/gtfs/rb_norway-aggregated-gtfs.zip")
        # https://storage.googleapis.com/marduk-production/outbound/gtfs/rb_atb-aggregated-gtfs.zip
        with open(Path(provider_folder, f"{localized_date_str}.zip"), 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(Path(provider_folder, f"{localized_date_str}.zip"), 'r') as zip_ref:
            destination_folder = Path(provider_folder, f"{localized_date_str}")
            destination_folder.mkdir(exist_ok=True, parents=True)
            zip_ref.extractall(destination_folder)
    except:
        print(f"ERROR downloading NWY static data")