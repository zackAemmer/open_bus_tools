import pickle
import pytz
import requests
from xml.etree import ElementTree

from dotenv import load_dotenv
import pandas as pd

import scrape_utils


if __name__ == "__main__":
    load_dotenv()
    # Limited to 4 requests/minute, otherwise need publish/subscribe
    endpoint = 'https://api.entur.io/realtime/v1/rest/vm'
    # Call Entur SIRI API (returns XML)
    response = requests.get(endpoint)
    root = ElementTree.fromstring(response.content)
    # Look at list of active vehicles from response
    root_dict = scrape_utils.xml_to_dict(root)
    data = root_dict['ServiceDelivery']['VehicleMonitoringDelivery']['VehicleActivity']
    # Remove inactive or non-monitored trips
    data = [x for x in data if 'MonitoredVehicleJourney' in x.keys()]
    data = [x for x in data if 'RecordedAtTime' in x.keys()]
    # Remove trips without required variables
    data = [x for x in data if 'FramedVehicleJourneyRef' in x['MonitoredVehicleJourney'].keys()]
    data = [x for x in data if 'DatedVehicleJourneyRef' in x['MonitoredVehicleJourney']['FramedVehicleJourneyRef'].keys()]
    data = [x for x in data if 'VehicleRef' in x['MonitoredVehicleJourney'].keys()]
    data = [x for x in data if 'DataSource' in x['MonitoredVehicleJourney'].keys()]
    data = [x for x in data if 'VehicleLocation' in x['MonitoredVehicleJourney'].keys()]

    # Move json to dataframe
    locationtimes = [str(scrape_utils.datetime_to_epoch(x['RecordedAtTime'])) for x in data]
    trip_ids = [str(x['MonitoredVehicleJourney']['FramedVehicleJourneyRef']['DatedVehicleJourneyRef']) for x in data]
    vehicle_ids = [str(x['MonitoredVehicleJourney']['VehicleRef']) for x in data]
    operator_ids = [str(x['MonitoredVehicleJourney']['DataSource']) for x in data]
    lat = [str(x['MonitoredVehicleJourney']['VehicleLocation']['Latitude']) for x in data]
    lons = [str(x['MonitoredVehicleJourney']['VehicleLocation']['Longitude']) for x in data]
    try:
        schedule_deviations = [str(x['MonitoredVehicleJourney']['Delay']) for x in data]
    except:
        schedule_deviations = ['' for x in data]
    data = pd.DataFrame({
        "trip_id": trip_ids,
        "vehicle_id": vehicle_ids,
        "operator_id": operator_ids,
        "lat": lat,
        "lon": lons,
        "scheduleDeviation": schedule_deviations,
        "locationtime": locationtimes
    }).sort_values(['trip_id','locationtime'])

    # Save scrape to file
    date_str, current_epoch, _ = scrape_utils.get_time_info(pytz.timezone("Europe/Oslo"))
    with open(f"./open_bus_tools/web_scraper/scraped_data/nwy/{date_str}_{current_epoch}.pkl", "wb") as f:
        pickle.dump(data, f)