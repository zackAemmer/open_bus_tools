import datetime as dt
from pathlib import Path
import pandas as pd

import boto3
from dotenv import load_dotenv
load_dotenv()


def download_new_s3_files(data_folder, bucket_name, uuid=None, date_range=None):
    print(f"Getting new files for {data_folder} from S3 bucket {bucket_name}")
    data_folder.mkdir(parents=True, exist_ok=True)
    downloaded_files = [x.name.split('.')[0] for x in data_folder.glob("*.pkl")]
    print(f"Found {len(downloaded_files)} downloaded files")
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        objs = list(bucket.objects.all())
        available_files = [o.key for o in objs]
        print(f"Successfully connected to S3")
        # Get list of files that are not already downloaded
        if uuid:
            available_files = [x for x in available_files if x.split('_')[-1].split('.')[0] == uuid]
        if date_range:
            available_files = [x for x in available_files if dt.datetime.strptime(date_range[0],"%Y_%m_%d") <= dt.datetime.strptime(x[:10],"%Y_%m_%d") <= dt.datetime.strptime(date_range[1],"%Y_%m_%d")]
        # Check which files are already downloaded
        new_files = [x for x in available_files if x[:10] not in downloaded_files]
        print(f"Found {len(new_files)} new files to download out of {len(available_files)} files in the specified bucket")
        # Download all new files to same data folder
        for i, file_name in enumerate(new_files):
            print(f"Downloading file {i} out of {len(new_files)}")
            bucket.download_file(file_name, Path(data_folder, f"{file_name[:10]}.pkl"))
    except ValueError:
        print(f"ERROR downloading {data_folder} files from S3 bucket {bucket_name}")


if __name__ == "__main__":
    print(f"Downloading new files...")
    download_new_s3_files(Path("./ExtremeSSD/data/kcm_realtime/"), "gtfsrt-collection-kcm", date_range=('2020_04_07', '2026_04_20'))
    download_new_s3_files(Path("./ExtremeSSD/data/atb_realtime/"), "gtfsrt-collection-atb", date_range=('2020_04_07', '2026_04_20'))
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        download_new_s3_files(Path('ExtremeSSD', 'data', 'other_feeds', f"{row['uuid']}_realtime"), 'gtfsrt-collection-others', uuid=row['uuid'], date_range=('2020_04_07', '2026_04_20'))