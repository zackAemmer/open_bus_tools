import datetime
from pathlib import Path
import pickle

import boto3
import gtfs_realtime_pb2
import pandas as pd
import pytz
from dotenv import load_dotenv
load_dotenv()


def summarize_realtime_positions(provider_folder, bucket_name):
    utc_date = datetime.datetime.now(pytz.timezone('utc'))
    summary_date = utc_date - datetime.timedelta(days=1)
    # Combine the protobuf files for each day/provider into a dataframe
    try:
        daily_veh_positions = {}
        # Each collected file gets combined for given provider
        for pb_file in provider_folder.glob('*.pb'):
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
                Bucket=bucket_name,
                Key=f"{summary_date.strftime('%Y_%m_%d')}_{provider_folder.name.split('_')[0]}.pkl"
            )
    except Exception as e:
        print('ERROR: ', provider_folder.name)
        print(e)


if __name__ == "__main__":
    summarize_realtime_positions(Path('open_bus_tools', 'webscraper', 'scraped_data', 'kcm_realtime', 'pbf'), 'gtfsrt-collection-kcm')
    summarize_realtime_positions(Path('open_bus_tools', 'webscraper', 'scraped_data', 'atb_realtime', 'pbf'), 'gtfsrt-collection-atb')
    for provider_folder in Path('open_bus_tools', 'webscraper', 'scraped_data', 'other_feeds').glob('*_realtime'):
        summarize_realtime_positions(Path('open_bus_tools', 'webscraper', 'scraped_data', 'other_feeds', provider_folder.name, 'pbf'), 'gtfsrt-collection-others')