#!/usr/bin/env python3
"""
Convert the .pb data/response files to .tfrecord files for use in TensorFlow

Creates state vectors at each device motion timestamp of the last sensor
values for all the other sensors (raw accel, location). Ignores battery
and skips magnetometer (not available on the watch).
"""
import os
import json
import socket
import pathlib
import urllib.request

from absl import app
from absl import flags
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError

from osm_tags import categories, types
from pool import run_job_pool
from decoding import parse_data, parse_response
from data_iterator import DataIterator
from writers import TFRecordWriter, CSVWriter
from state_vector_peek_iterator import StateVectorPeekIterator

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", None, "Directory where the watch files are (can be in subdirs of this)")
flags.DEFINE_integer("jobs", 0, "How many jobs to run simultaneously (0 = number of cores)")
flags.DEFINE_string("nums", None, "Comma-separated list of which watch numbers to process (e.g. 1,2,3,4,...,15)")
flags.DEFINE_integer("window_size", 50, "Size of window to use to make sure the file's out-of-order samples are sorted, done per-sensor")
flags.DEFINE_integer("begin_offset", -60, "Seconds offset to start looking at data after (before is negative) the label timestamp")
flags.DEFINE_integer("end_offset", -30, "Seconds offset to end looking at data after (before is negative) the label timestamp")
flags.DEFINE_enum("output", "tfrecord", ["tfrecord", "csv"], "Output format to save windows using")
flags.DEFINE_boolean("debug", False, "Print debug information")

flags.mark_flag_as_required("dir")
flags.mark_flag_as_required("nums")


def get_watch_files(watch_number, responses=False):
    """
    For a directory FLAGS.dir, we should get all the *.pb files in any subfolder
    that has the name watchXYZ for XYZ=watch_number with leading zeros to make
    it exactly 3 characters. For example, if we have subdirectories week{1,2,3},
    then we would match:
        dir/week1/watch001/sensor_data_*.pb
        dir/week2/watch001/sensor_data_*.pb
        dir/week3/watch001/sensor_data_*.pb

    If responses=True, then we instead match responses_*.pb in each directory.
    """
    name = "watch%03d"%watch_number

    if responses:
        prefix = "responses_"
    else:
        prefix = "sensor_data_"

    files = list(pathlib.Path(FLAGS.dir).glob("**/%s/%s*"%(name, prefix)))

    # Sort so we get them in time ascending order
    files.sort()

    # Return as a string rather than a PosixPath so we can directly pass
    # to open(..., 'b')
    return [str(x) for x in files]


def reverse_geocode(lat, lon, timeout=300):
    """
    Get location information about a GPS coordinate (lat, lon) with
    a local instance of Open Street Maps Nominatim

    This assumes you have a Nominatim server running on 7070 as described
    in README.md

    We use a large timeout since sometimes postgresql decides to autovacuum the
    database which takes a bit of time. Though, really, we should probably just
    wait till it's done doing that.

    We could use geopy, but it seems to download this in the "json" format
    hard-coded, but we need category, etc. information only available in the
    other formats such as jsonv2.

    See hard-coded format in geopy:
    https://github.com/geopy/geopy/blob/85ccae74f2e8011c89d2e78d941b5df414ab99d1/geopy/geocoders/osm.py#L453

    See example reverse lookup JSON with "jsonv2" format:
    https://nominatim.org/release-docs/develop/api/Reverse/#example-with-formatjsonv2

    Code for handling timeouts:
    https://stackoverflow.com/q/8763451
    """
    url = "http://localhost:7070/reverse?format=jsonv2&lat=" \
        + str(lat) + "&lon=" + str(lon)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as con:
            return json.loads(con.read().decode("utf-8"))
    except HTTPError as error:
        print("Warning:", error, "loading", url)
    except URLError as error:
        if isinstance(error.reason, socket.timeout):
            print("Warning: socket timed out loading", url)
        else:
            print("Warning: unknown error loading", url)
    except socket.timeout:
        print("Warning: socket timed out loading", url)

    return None


def one_hot_location(possible_values, value, list_name=None):
    """ Generate the one-hot vector for the OSM location categories/types """
    # Last one is a not-in-list "other" category/type
    results = [0]*(len(possible_values) + 1)

    if value in possible_values:
        results[possible_values.index(value)] = 1
    else:
        results[-1] = 1

        # For debugging, print out what (common?) values we missed
        if list_name is not None:
            print("Warning:", value, "not in", list_name)

    return results


def parse_state_vector(epoch, dm, acc, loc):
    """ Parse here rather than in DataIterator since we end up skipping lots
    of data, so if we do it now, we'll run the parsing way fewer times

    Time features:
        - second (/60)
        - minute (/60)
        - hour (/12)
        - hour (/24)
        - second of day (/86400)
        - day of week (/7)
        - day of month (/31)
        - day of year (/366)
        - month of year (/12)
        - year

    Reverse geocoded location - one-hot encoded, e.g. if we had 3 categories:
        <1,0,0,0> - amenity
        <0,1,0,0> - barrier
        <0,0,1,0> - bridge
        <0,0,0,1> - "other"
    but for all the location categories and types in osm_tags.py. Additional
    "other" category/type if not in the list of possible categories/types.

    Then, followed by the raw data.
    """
    if dm is not None:
        dm = parse_data(dm)
    if acc is not None:
        acc = parse_data(acc)
    if loc is not None:
        loc = parse_data(loc)

    time_features = [
        epoch.second,
        epoch.minute,
        epoch.hour % 12,  # 12-hour
        epoch.hour,  # 24-hour
        (epoch - epoch.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds(),  # since midnight
        epoch.weekday(),
        epoch.day,  # day of month
        (epoch - epoch.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)).days + 1,  # day of year
        epoch.month,
        epoch.year
    ]

    raw_data = [
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
        dm["heading"] if dm is not None else None,
        acc["raw_acceleration"]["x"] if acc is not None else None,
        acc["raw_acceleration"]["y"] if acc is not None else None,
        acc["raw_acceleration"]["z"] if acc is not None else None,
        loc["longitude"] if loc is not None else None,
        loc["latitude"] if loc is not None else None,
        loc["horizontal_accuracy"] if loc is not None else None,
        loc["altitude"] if loc is not None else None,
        loc["vertical_accuracy"] if loc is not None else None,
        loc["course"] if loc is not None else None,
        loc["speed"] if loc is not None else None,
        loc["floor"] if loc is not None else None,
    ]

    # Default location to an additional "other" type of location. If we can
    # do the reverse lookup, it'll instead fill in the appropriate category/type
    location_category_features = [0]*len(categories) + [1]
    location_type_features = [0]*len(types) + [1]

    if loc is not None:
        location = reverse_geocode(loc["latitude"], loc["longitude"])

        if location is not None:
            location_category_features = one_hot_location(categories,
                location["category"], "categories")
            location_type_features = one_hot_location(types,
                location["type"], "types")

            #print("found", location["category"], location["type"], "for",
            #    str(loc["latitude"])+", "+str(loc["longitude"]))

    features = time_features + location_category_features \
        + location_type_features + raw_data

    return features


def process_watch(watch_number):
    """
    Process the data from one watch from any week (matches multiple subdirs)

    Note: this stores the results entirely in memory till we write to disk at
    the end so we can do stuff like train/test splits, split into windows, etc.
    Will need to be tweaked if at some point one users's data doesn't entirely
    fit into memory. (Not sure sklearn.model_selection.train_test_split works
    for that?)
    """
    # Get the data files and response files for this watch number
    data_files = get_watch_files(watch_number)
    response_files = get_watch_files(watch_number, responses=True)

    # We'll create a state vector for the appropriate time windows nearby the
    # labeled responses, so create iterator over responses and then in the
    # state vector iterator, we'll create the iterators over the data.
    respIter = DataIterator(response_files, responses=True,
        window=FLAGS.window_size)
    statePeekIter = StateVectorPeekIterator(data_files, window=FLAGS.window_size)

    # Generate window for each response
    windows = []
    printed_feature_count = False

    for resp in respIter:
        beginTime = datetime.fromtimestamp(resp.epoch) \
            + timedelta(seconds=FLAGS.begin_offset)
        endTime = datetime.fromtimestamp(resp.epoch) \
            + timedelta(seconds=FLAGS.end_offset)

        # Skip state vectors up till the begin time, then stop when we
        # reach end time. Note: this is a "peek" iterator -- if the state's
        # value is beyond the endTime and will be potentially used in the next
        # window, then don't consume/pop it. Otherwise, though, pop the
        # state so that the next iterator call will be different (not inf loop).
        x = []

        for epoch, dm, acc, loc in statePeekIter:
            epoch = datetime.fromtimestamp(epoch)

            if epoch < beginTime:
                statePeekIter.pop()
                continue
            elif epoch <= endTime:
                x.append(parse_state_vector(epoch, dm, acc, loc))
                statePeekIter.pop()

                # Print out the number of features once
                if not printed_feature_count:
                    print("Watch%03d"%watch_number + ": Num features:", len(x[-1]))
                    printed_feature_count = True
            else:
                # Don't consume - we'll look at this state for the next
                # window
                break

        # Save this window if we have data for it
        y = parse_response(resp)["label"]

        if len(x) > 0:
            if FLAGS.debug:
                print("Watch%03d"%watch_number + ": found data for label",
                    y, "at time", str(datetime.fromtimestamp(resp.epoch)),
                    "having", len(x), "time steps")
            windows.append((x, y))
        elif FLAGS.debug:
            print("Watch%03d"%watch_number + ": Warning: no data for label",
                y, "at time", str(datetime.fromtimestamp(resp.epoch)))

    # Output
    if FLAGS.output == "tfrecord":
        # Write both including the other class and not including it
        writer = TFRecordWriter(watch_number, include_other=True)
        writer.write_records(windows)

        writer = TFRecordWriter(watch_number, include_other=False)
        writer.write_records(windows)
    elif FLAGS.output == "csv":
        writer = CSVWriter(watch_number)
        writer.write_records(windows)
    else:
        raise NotImplementedError("unsupported output format: "+FLAGS.output)


def main(argv):
    assert FLAGS.end_offset - FLAGS.begin_offset > 0, \
        "end_offset - begin_offset should be positive"

    watch_numbers = [int(x) for x in FLAGS.nums.split(",")]

    # Process all the provided watch numbers
    if FLAGS.jobs != 1:
        # Make TensorFlow ignore the GPU since we run many jobs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if FLAGS.jobs == 0:
            cores = None
        else:
            cores = FLAGS.jobs

        run_job_pool(process_watch, [(d,) for d in watch_numbers], cores=cores)
    else:
        for watch_number in watch_numbers:
            process_watch(watch_number)


if __name__ == "__main__":
    app.run(main)
