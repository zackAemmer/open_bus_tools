import os

import boto3
from dotenv import load_dotenv
load_dotenv()

from openbustools import standardfeeds


def download_new_s3_files(data_folder, bucket_name):
    print(f"Getting new files for {data_folder} from S3 bucket {bucket_name}")
    downloaded_files = os.listdir(data_folder)
    print(f"Found {len(downloaded_files)} downloaded files")
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        objs = list(bucket.objects.all())
        available_files = [o.key for o in objs]
        print(f"Successfully connected to S3")
        # Get list of files that are not already downloaded
        new_files = [x for x in available_files if x not in downloaded_files]
        print(f"Found {len(new_files)} new files to download out of {len(available_files)} files in the specified bucket")
        # Download all new files to same data folder
        for i,file in enumerate(new_files):
            print(f"Downloading file {i} out of {len(new_files)}")
            bucket.download_file(f"{file}", f"{data_folder}{file}")
    except ValueError:
        print(f"Failure to access S3")
    return None


if __name__ == "__main__":
    print(f"Downloading new files...")
    download_new_s3_files("./data/kcm_realtime/", "gtfs-collection-kcm")
    download_new_s3_files("./data/nwy_realtime/", "gtfs-collection-nwy")
    print(f"Extracting operators from downloaded files...")
    standardfeeds.extract_operator("./data/nwy_realtime/", "./data/atb_realtime/", "operator_id", "ATB")
    standardfeeds.extract_operator("./data/nwy_realtime/", "./data/rut_realtime/", "operator_id", "RUT")
    print(f"Extracting operators from GTFS files...")
    standardfeeds.extract_operator_gtfs("./data/nwy_gtfs/", "./data/atb_gtfs/", "trip_id", "trip_id", "ATB")
    standardfeeds.extract_operator_gtfs("./data/nwy_gtfs/", "./data/rut_gtfs/", "trip_id", "trip_id", "RUT")