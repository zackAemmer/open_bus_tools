from pathlib import Path

import boto3
from dotenv import load_dotenv
load_dotenv()

from openbustools import spatial, standardfeeds


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
        # new_files = [x for x in available_files if x not in downloaded_files]
        new_files = standardfeeds.get_date_list('2024_03_15', 3)
        print(f"Found {len(new_files)} new files to download out of {len(available_files)} files in the specified bucket")
        # Download all new files to same data folder
        for i, file_name in enumerate(new_files):
            try:
                print(f"Downloading file {i} out of {len(new_files)}")
                bucket.download_file(file_name, Path(data_folder, file_name))
            except:
                print(f"Error for file {file_name}")
                continue
    except ValueError:
        print(f"Failure to access S3")
    return None


if __name__ == "__main__":
    print(f"Downloading new files...")
    download_new_s3_files("./data/kcm_realtime/", "gtfs-collection-kcm")
    # download_new_s3_files("./data/nwy_realtime/", "gtfs-collection-nwy")
    # print(f"Extracting operators from downloaded realtime and static files...")
    # area = spatial.make_polygon((10.01266280018279,63.241039487344544,10.604534521465991,63.475046970112395))
    # standardfeeds.extract_operator_gtfs(Path("data", "nwy_static"), Path("data", "atb_static"), area)
    # standardfeeds.extract_operator("./data/nwy_realtime/", "./data/atb_realtime/", "operator_id", "ATB")