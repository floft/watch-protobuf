"""
Create features from raw data and labels
"""
from datetime import datetime
from geocode import reverse_geocode
from osm_tags import categories, types

from decoding import parse_data, parse_response


def one_hot_location(possible_values, value):
    """ Generate the one-hot vector for the OSM location categories/types """
    # Last one is a not-in-list "other" category/type
    results = [0]*(len(possible_values) + 1)

    if value in possible_values:
        results[possible_values.index(value)] = 1
    else:
        results[-1] = 1

    return results


def create_time_features(epoch):
    """
    Time features:
        - second (/60)
        - minute (/60)
        - hour (/12)
        - hour (/24)
        - second of day (/86400)
        - day of week (/7)

    Skip for now... won't generalize well due to short experiments. Probably
    would cause problems when normalizing since we have very small ranges of
    these in the training data. Note: non-full-data experiments had these in
    them.
        - day of month (/31)
        - day of year (/366)
        - month of year (/12)
        - year
    """
    # Convert to datetime object if not one already
    if not isinstance(epoch, datetime):
        epoch = datetime.fromtimestamp(epoch)

    return [
        epoch.second,
        epoch.minute,
        epoch.hour % 12,  # 12-hour
        epoch.hour,  # 24-hour
        (epoch - epoch.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds(),  # since midnight
        epoch.weekday(),

        # epoch.day,  # day of month
        # (epoch - epoch.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)).days + 1,  # day of year
        # epoch.month,
        # epoch.year
    ]


def create_dm_features(dm, time=False):
    """ Raw data """
    time_features = []
    if time:
        time_features = create_time_features(dm["epoch"])

    return time_features + [
        dm["attitude"]["roll"] if dm is not None else None,
        dm["attitude"]["pitch"] if dm is not None else None,
        dm["attitude"]["yaw"] if dm is not None else None,
        dm["rotation_rate"]["x"] if dm is not None else None,
        dm["rotation_rate"]["y"] if dm is not None else None,
        dm["rotation_rate"]["z"] if dm is not None else None,
        dm["user_acceleration"]["x"] if dm is not None else None,
        dm["user_acceleration"]["y"] if dm is not None else None,
        dm["user_acceleration"]["z"] if dm is not None else None,
        dm["gravity"]["x"] if dm is not None else None,
        dm["gravity"]["y"] if dm is not None else None,
        dm["gravity"]["z"] if dm is not None else None,
        #dm["heading"] if dm is not None else None,
    ]


def create_acc_features(acc, time=False):
    """ Raw data """
    time_features = []
    if time:
        time_features = create_time_features(acc["epoch"])

    return time_features + [
        acc["raw_acceleration"]["x"] if acc is not None else None,
        acc["raw_acceleration"]["y"] if acc is not None else None,
        acc["raw_acceleration"]["z"] if acc is not None else None,
    ]


def create_loc_features(loc, location_categories=None, time=False):
    """ Raw data followed by location category/type features

    Reverse geocoded location - one-hot encoded, e.g. if we had 3 categories:
        <1,0,0,0> - amenity
        <0,1,0,0> - barrier
        <0,0,1,0> - bridge
        <0,0,0,1> - "other"
    but for all the location categories and types in osm_tags.py. Additional
    "other" category/type if not in the list of possible categories/types.
    """
    time_features = []
    if time:
        time_features = create_time_features(loc["epoch"])

    # We throw out the lat, long, etc. features regarding location since we
    # instead add the category/type reverse geocode information.
    # Throw out heading - I think that's from the magnetometer, which we don't
    # have on this watch.
    raw_loc_features = [
        #loc["longitude"] if loc is not None else None,
        #loc["latitude"] if loc is not None else None,
        #loc["horizontal_accuracy"] if loc is not None else None,
        loc["altitude"] if loc is not None else None,
        #loc["vertical_accuracy"] if loc is not None else None,
        loc["course"] if loc is not None else None,
        loc["speed"] if loc is not None else None,
        #loc["floor"] if loc is not None else None,
    ]

    # Default location to an additional "other" type of location. If we can
    # do the reverse lookup, it'll instead fill in the appropriate category/type
    location_category_features = [0]*len(categories) + [1]
    location_type_features = [0]*len(types) + [1]

    if loc is not None:
        location = reverse_geocode(loc["latitude"], loc["longitude"])

        if location is not None:
            location_category_features = one_hot_location(categories,
                location["category"])
            location_type_features = one_hot_location(types,
                location["type"])

            # Keep track of how many there were of each
            if location_categories is not None:
                location_categories[(location["category"], location["type"])] += 1

            #print("found", location["category"], location["type"], "for",
            #    str(loc["latitude"])+", "+str(loc["longitude"]))

    return time_features + raw_loc_features + location_category_features \
        + location_type_features


def parse_state_vector(epoch, dm, acc, loc, location_categories=None):
    """ Parse here rather than in DataIterator since we end up skipping lots
    of data, so if we do it now, we'll run the parsing way fewer times """
    if dm is not None:
        dm = parse_data(dm)
    if acc is not None:
        acc = parse_data(acc)
    if loc is not None:
        loc = parse_data(loc)

    time_features = create_time_features(epoch)
    dm_features = create_dm_features(dm)
    acc_features = create_acc_features(acc)
    loc_features = create_loc_features(loc, location_categories)

    return time_features + dm_features + acc_features + loc_features


def parse_response_vector(resp, time=False):
    resp = parse_response(resp)

    # For process_full.py we'll keep the timestamp and label??? TODO
    if time:
        time_features = create_time_features(resp["epoch"])
        return time_features + [resp["label"]]

    # For process.py, we want an int, not a vector
    else:
        return resp["label"]


def parse_full_data(window, location_categories=None):
    """ Parse here rather than in DataIterator since we end up skipping lots
    of data, so if we do it now, we'll run the parsing way fewer times """
    dm = window.dm
    acc = window.acc
    loc = window.loc
    # resp = window.resp

    assert dm is not None
    assert acc is not None
    assert loc is not None
    # assert resp is not None

    dm_features = [create_dm_features(parse_data(x), time=True) for x in dm]
    acc_features = [create_acc_features(parse_data(x), time=True) for x in acc]
    loc_features = [create_loc_features(parse_data(x), location_categories, time=True) for x in loc]

    # Skip for now? TODO
    # resp_features = [parse_response_vector(x, time=True) for x in resp]

    return dm_features, acc_features, loc_features
