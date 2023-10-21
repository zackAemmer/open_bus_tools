from datetime import datetime, timedelta
import pickle
import requests
import sys
sys.path.append("../")

from dotenv import load_dotenv
load_dotenv()
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
    # Query OBA API for King County Metro
    endpoint = 'http://api.pugetsound.onebusaway.org/api/where/vehicles-for-agency/1.json?key='+secret.OBA_API_KEY
    response = requests.get(endpoint)
    # Filter as much as possible
    data = response.json()['data']['list']
    # Remove inactive or non-monitored trips
    data = [x for x in data if 'tripId' in x.keys()]
    data = [x for x in data if 'lastLocationUpdateTime' in x.keys()]
    data = [x for x in data if x['tripId']!='']
    data = [x for x in data if x['lastLocationUpdateTime']!=0]
    # Remove trips without required variables
    data = [x for x in data if 'lastLocationUpdateTime' in x.keys()]
    data = [x for x in data if 'tripStatus' in x.keys()]
    data = [x for x in data if 'activeTripId' in x['tripStatus'].keys()]
    data = [x for x in data if 'vehicleId' in x.keys()]
    data = [x for x in data if 'location' in x.keys()]
    data = [x for x in data if 'orientation' in x['tripStatus'].keys()]

    # Move from json to dataframe
    locationtimes = [str(x['lastLocationUpdateTime'])[:-3] for x in data]
    trip_ids = [str(x['tripStatus']['activeTripId'])[2:] for x in data]
    vehicle_ids = [str(x['vehicleId'])[2:] for x in data]
    lat = [str(x['location']['lat']) for x in data]
    lons = [str(x['location']['lon']) for x in data]
    orientations = [str(x['tripStatus']['orientation']) for x in data]
    try:
        schedule_deviations = [str(x['tripStatus']['scheduleDeviation']) for x in data]
    except:
        schedule_deviations = ['' for x in data]
    try:
        next_stops = [str(x['tripStatus']['nextStop'])[2:] for x in data]
    except:
        next_stops = ['' for x in data]
    data = pd.DataFrame({
        "trip_id": trip_ids,
        "vehicle_id": vehicle_ids,
        "lat": lat,
        "lon": lons,
        "orientation": orientations,
        "scheduleDeviation": schedule_deviations,
        "nextStop": next_stops,
        "locationtime": locationtimes
    }).sort_values(['trip_id','locationtime'])
    
    # Save scrape to file
    date_str, current_epoch = get_time_info(-8)
    with open(f"./scraped_data/kcm/{date_str}_{current_epoch}.pkl", "wb") as f:
        pickle.dump(data, f)