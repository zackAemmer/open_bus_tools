from pathlib import Path

import pandas as pd

import gtfs_realtime_pb2


if __name__ == "__main__":
    # Combine the protobuf files for each day/provider into a dataframe
    for provider_folder in Path('data', 'other_feeds').glob('*_realtime'):
        try:
            daily_veh_positions = {}
            for pb_file in Path(provider_folder, 'pbf').glob('*.pb'):
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
            try:
                daily_df = pd.DataFrame.from_dict(daily_veh_positions, orient='index').drop_duplicates().sort_values(['trip_id', 'locationtime'])
                daily_df.to_pickle(Path(provider_folder, f"{pb_file.name[:10]}.pkl"))
            except:
                print('ERROR saving: ', provider_folder.name)
                continue
        except:
            print('ERROR loading: ', provider_folder.name)
            continue