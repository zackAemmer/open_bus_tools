from datetime import datetime
import os
import pickle

import boto3
from dotenv import load_dotenv
import pandas as pd


def get_time_info(tz=None):
    current_target_time = datetime.now(tz)
    date_str = current_target_time.strftime("%Y_%m_%d_%H")
    epoch = round(current_target_time.timestamp())
    return date_str, epoch


if __name__ == "__main__":
    load_dotenv()
    # Load data
    scrape_folder = "./open_bus_tools/web_scraper/scraped_data/nwy/"
    scrape_files = os.listdir(scrape_folder)
    all_data = []
    date_str, current_epoch = get_time_info(1)
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
        date_str, current_epoch = get_time_info(1)
        s3 = boto3.client('s3')
        s3.put_object(
            Body=pickle.dumps(all_data),
            Bucket="gtfs-collection-nwy",
            Key=date_str[:10]+".pkl"
        )
    except:
        print(f"Either no files found for {date_str}, or failure to access S3")
