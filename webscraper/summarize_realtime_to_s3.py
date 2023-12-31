import boto3
import datetime
from dotenv import load_dotenv
from pathlib import Path
import pickle

import pandas as pd

import gtfs_realtime_pb2


if __name__ == "__main__":
    load_dotenv()
    utc_date = datetime.datetime.utcnow()
    summary_date = utc_date - datetime.timedelta(days=1)
    # Combine the protobuf files for each day/provider into a dataframe
    for provider_folder in Path('data', 'other_feeds').glob('*_realtime'):
        try:
            daily_veh_positions = {}
            # Each collected file gets combined for given provider
            for pb_file in Path(provider_folder, 'pbf').glob('*.pb'):
                # Check that file is from previous day
                if pb_file.name[:10] == summary_date.strftime('%Y_%m_%d'):
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
                    # Remove the pb file
                    pb_file.unlink()
            # If there were no records for the day, skip
            if len(daily_veh_positions) > 0:
                daily_df = pd.DataFrame.from_dict(daily_veh_positions, orient='index').drop_duplicates().sort_values(['trip_id', 'locationtime'])
                # daily_df.to_pickle(Path(provider_folder, f"{pb_file.name[:10]}.pkl"))
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