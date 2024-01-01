from pathlib import Path
import pandas as pd

import boto3
from dotenv import load_dotenv
load_dotenv()


def download_new_s3_files(data_folder, bucket_name):
    print(f"Getting new files for {data_folder} from S3 bucket {bucket_name}")
    downloaded_files = [x.name for x in Path(data_folder).glob("*.pkl")]
    print(f"Found {len(downloaded_files)} downloaded files")
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        objs = list(bucket.objects.all())
        available_files = [o.key for o in objs]
        print(f"Successfully connected to S3")
        # Get list of files that are not already downloaded
        available_files = [x for x in available_files if x.split('_')[-1].split('.')[0] == data_folder.name.split('_')[0]]
        new_files = [x for x in available_files if f"{x[:10]}.pkl" not in downloaded_files]
        print(f"Found {len(new_files)} new files to download out of {len(available_files)} files in the specified bucket")
        # Download all new files to same data folder
        for i, file_name in enumerate(new_files):
            print(f"Downloading file {i} out of {len(new_files)}")
            bucket.download_file(file_name, Path(data_folder, f"{file_name[:10]}.pkl"))
    except ValueError:
        print(f"Failure to access S3")


if __name__ == "__main__":
    print(f"Downloading new files...")
    cleaned_sources = pd.read_csv(Path('data', 'cleaned_sources.csv'))
    for i, row in cleaned_sources.iterrows():
        download_new_s3_files(Path('data', 'other_feeds', f"{row['uuid']}_realtime"), 'gtfs-collection-others')