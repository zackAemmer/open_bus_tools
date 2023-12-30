import datetime
from pathlib import Path
import zipfile

import requests
import numpy as np
import pandas as pd
import pytz


if __name__ == "__main__":
    # Load cleaned list of transit data providers
    cleaned_sources = pd.read_csv(Path('data', 'other_feeds', 'cleaned_sources.csv'))
    # For each transit feed, query static data, unzip and save to folder
    for index, row in cleaned_sources.iterrows():
        print(row['provider'])
        # provider_folder = Path('data', 'other_feeds', f"{row['uuid']}_static")
        # provider_folder.mkdir(exist_ok=True, parents=True)
        # localized_date = datetime.datetime.now().astimezone(pytz.timezone(row['tz_str']))
        # localized_date_str = localized_date.strftime("%Y_%m_%d")
        # try:
        #     response = requests.get(row['static_url'])
        #     with open(Path(provider_folder, f"{localized_date_str}.zip"), 'wb') as f:
        #         f.write(response.content)
        #     with zipfile.ZipFile(Path(provider_folder, f"{localized_date_str}.zip"), 'r') as zip_ref:
        #         destination_folder = Path(provider_folder, f"{localized_date_str}")
        #         destination_folder.mkdir(exist_ok=True, parents=True)
        #         zip_ref.extractall(destination_folder)
        # except:
        #     print(f"ERROR downloading {row['provider']} static data")
        #     continue