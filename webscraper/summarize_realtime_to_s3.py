import boto3
import datetime
from dotenv import load_dotenv
from pathlib import Path
import pickle

import pandas as pd

import gtfs_realtime_pb2


if __name__ == "__main__":
    # When feeds are removed from the cleaned_sources.csv, the realtime data will stop being collected
    # When this script runs it will empty those folders to avoid saving data and filling up the disk, but cease to refill them
    # Then those files should be removed from the S3 bucket (manually for now, could be worth script later)
    load_dotenv()
    utc_date = datetime.datetime.utcnow()
    summary_date = utc_date - datetime.timedelta(days=1)
    # Combine the protobuf files for each day/provider into a dataframe
    for provider_folder in Path('open_bus_tools', 'data', 'other_feeds').glob('*_realtime'):
        # Wrap each feed in try loop
        try:
            daily_veh_positions = {}
            # Each collected file gets combined for given provider
            for pb_file in Path(provider_folder, 'pbf').glob('*.pb'):
                # Check that file is from previous day
                pb_file_datetime = datetime.datetime.strptime(pb_file.name[:10], '%Y_%m_%d')
                if pb_file_datetime.date() == summary_date.date():
                    # Wrap each tracked time in try loop
                    try:
                        with open(pb_file, 'rb') as f:
                            veh_positions = gtfs_realtime_pb2.FeedMessage()
                            veh_positions.ParseFromString(f.read())
                            for entity in veh_positions.entity:
                                if entity.HasField('vehicle'):
                                    daily_veh_positions[f"{pb_file.name}_{entity.id}"] = {
                                        'trip_id': entity.vehicle.trip.trip_id,
                                        'file': pb_file.name[:10],
                                        'locationtime': entity.vehicle.timestamp,
                                        'lat': entity.vehicle.position.latitude,
                                        'lon': entity.vehicle.position.longitude,
                                        'vehicle_id': entity.vehicle.vehicle.id
                                    }
                    except Exception as e:
                        continue
                    # Remove the tracked time file regardless if it was successfully read
                    pb_file.unlink()
                # Remove the tracked time file if it is from a previous date
                elif pb_file_datetime.date() < summary_date.date():
                    pb_file.unlink()
            # If there were no records for the day, skip
            if len(daily_veh_positions) > 0:
                daily_df = pd.DataFrame.from_dict(daily_veh_positions, orient='index').drop_duplicates().sort_values(['trip_id', 'locationtime'])
                # Upload to S3
                s3 = boto3.client('s3')
                s3.put_object(
                    Body=pickle.dumps(daily_df),
                    Bucket="gtfs-collection-others",
                    Key=f"{summary_date.strftime('%Y_%m_%d')}_{provider_folder.name.split('_')[0]}.pkl"
                )
        except Exception as e:
            print('ERROR: ', provider_folder.name)
            print(e)
            continue