from datetime import datetime, timedelta
import os
import pickle

import boto3
from dotenv import load_dotenv
import pandas as pd


def get_time_info(time_delta=0):
    # Get the UTC
    utc = datetime.utcnow()
    adj = timedelta(hours=time_delta)
    target_time = (utc + adj)
    date_str = target_time.strftime("%Y_%m_%d_%H")
    epoch = round(utc.timestamp())
    return date_str, epoch


if __name__ == "__main__":
    load_dotenv()
    # Load data
    scrape_folder = "./open_bus_tools/web_scraper/scraped_data/kcm/"
    scrape_files = os.listdir(scrape_folder)
    all_data = []
    date_str, current_epoch = get_time_info(-8)
    for filename in scrape_files:
        if filename[-4:]==".pkl" and filename[:10]!=date_str[:10]:
            with open(scrape_folder+filename, 'rb') as f:
                data = pickle.load(f)
            all_data.append(data)
            os.remove(scrape_folder+filename)
    # Combine and remove duplicated locations
    try:
        all_data = pd.concat(all_data)
        all_data = all_data.drop_duplicates(['trip_id','locationtime']).sort_values(['trip_id','locationtime'])
        # Upload to S3
        date_str, current_epoch = get_time_info(-8)
        s3 = boto3.client('s3')
        s3.put_object(
            Body=pickle.dumps(all_data),
            Bucket="gtfs-collection-kcm",
            Key=date_str[:10]+".pkl"
        )
    except ValueError:
        print(f"Either no files found for {date_str}, or failure to access S3")