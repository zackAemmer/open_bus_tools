from datetime import datetime, timedelta
import pickle
import pytz
import re
import requests
from xml.etree import ElementTree

from dotenv import load_dotenv
import pandas as pd


def get_time_info(time_delta=0):
    # Get the UTC
    utc = datetime.utcnow()
    adj = timedelta(hours=time_delta)
    target_time = (utc + adj)
    date_str = target_time.strftime("%Y_%m_%d_%H")
    epoch = round(utc.timestamp())
    return date_str, epoch

def datetime_to_epoch(time):
    # Go from time string in CET to epoch
    # format is '2022-03-16T16:14:58.12+01:00'
    yr = int(time[0:4])
    mo = int(time[5:7])
    day = int(time[8:10])
    hr = int(time[11:13])
    mn = int(time[14:16])
    sec = int(time[17:19])
    timezone = pytz.timezone('Europe/Oslo')
    ts = datetime(year=yr, month=mo, day=day, hour=hr, minute=mn, second=sec)
    ts = timezone.localize(ts)
    return ts.timestamp()

def xml_to_dict(element):
    # Recursively create a dictionary of XML field -> text value
    element_dict = {}
    for child in element:
        tag = re.split("}", child.tag)[1]
        if child.text != None:
            element_dict[tag] = child.text
        elif tag in element_dict.keys(): # In case multiple children with same tag exist in this element, turn into a list
            if type(element_dict[tag]) == list:
                element_dict[tag].append(xml_to_dict(child))
            else:
                first_elem = element_dict[tag]
                element_dict[tag] = []
                element_dict[tag].append(first_elem)
        else:
            element_dict[tag] = xml_to_dict(child)
    return element_dict


if __name__ == "__main__":
    load_dotenv()
    # Limited to 4 requests/minute, otherwise need publish/subscribe
    endpoint = 'https://api.entur.io/realtime/v1/rest/vm'
    # Call Entur SIRI API (returns XML)
    response = requests.get(endpoint)
    root = ElementTree.fromstring(response.content)
    # Look at list of active vehicles from response
    root_dict = xml_to_dict(root)
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
    locationtimes = [str(datetime_to_epoch(x['RecordedAtTime'])) for x in data]
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
    date_str, current_epoch = get_time_info(1)
    with open(f"./web_scraper/scraped_data/nwy/{date_str}_{current_epoch}.pkl", "wb") as f:
        pickle.dump(data, f)