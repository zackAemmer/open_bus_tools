from datetime import datetime
import re

import pytz


def get_time_info(tz=None):
    current_target_time = datetime.now(tz)
    date_str = current_target_time.strftime("%Y_%m_%d_%H")
    epoch = round(current_target_time.timestamp())
    return date_str, epoch, current_target_time


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