import os
import pickle
import pytz
import requests

from dotenv import load_dotenv
import pandas as pd

import scrape_utils


if __name__ == "__main__":
    load_dotenv()
    # Query OBA API for King County Metro
    endpoint = 'http://api.pugetsound.onebusaway.org/api/where/vehicles-for-agency/1.json?key='+os.getenv("OBA_API_KEY")
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
    date_str, current_epoch, _ = scrape_utils.get_time_info(pytz.timezone("America/Los_Angeles"))
    with open(f"./open_bus_tools/web_scraper/scraped_data/kcm/{date_str}_{current_epoch}.pkl", "wb") as f:
        pickle.dump(data, f)