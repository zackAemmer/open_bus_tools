import datetime
import os
import pickle

import boto3
from dotenv import load_dotenv
import pandas as pd
import pytz

import scrape_utils


if __name__ == "__main__":
    load_dotenv()
    scrape_folder = "./open_bus_tools/webscraper/scraped_data/kcm/"
    scrape_files = os.listdir(scrape_folder)
    all_data = []
    current_date_str, current_epoch, current_datetime = scrape_utils.get_time_info(pytz.timezone("America/Los_Angeles"))
    prev_datetime = current_datetime - datetime.timedelta(days=1)
    prev_date_str = prev_datetime.strftime("%Y_%m_%d_%H")
    # Load into memory then clear on disk any collection that occurred before the current day
    for filename in scrape_files:
        if filename[-4:]==".pkl" and filename[:10]!=current_date_str[:10]:
            with open(scrape_folder+filename, 'rb') as f:
                data = pickle.load(f)
            all_data.append(data)
            os.remove(scrape_folder+filename)
    # Combine and remove duplicated locations
    try:
        all_data = pd.concat(all_data)
        all_data = all_data.drop_duplicates(['trip_id','locationtime']).sort_values(['trip_id','locationtime'])
        # Upload to S3
        s3 = boto3.client('s3')
        s3.put_object(
            Body=pickle.dumps(all_data),
            Bucket="gtfs-collection-kcm",
            Key=prev_date_str[:10]+".pkl"
        )
    except Exception as e:
        print(f"Either no files found for {current_date_str}, or failure to access S3")
        print(e)