import datetime
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import pytz


if __name__ == "__main__":
    # Load cleaned list of transit data providers
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    # For each transit feed, query realtime data and save to folder
    for index, row in cleaned_sources.iterrows():
        provider_folder = Path('data', 'other_feeds', f"{row['uuid']}_realtime", 'pbf')
        provider_folder.mkdir(exist_ok=True, parents=True)
        try:
            current_target_time = datetime.datetime.now(pytz.timezone(row['tz_str']))
            current_epoch = round(current_target_time.timestamp())
            response = requests.get(row['realtime_url'])
            with open(Path(provider_folder, f"{current_target_time.strftime('%Y_%m_%d_%H')}_{current_epoch}.pb"), 'wb') as f:
                f.write(response.content)
        except:
            print(f"ERROR downloading {row['provider']} realtime data")
            continue