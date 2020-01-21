#!/usr/bin/env python3
"""
Convert the .pb data/response files to .tfrecord files for use in TensorFlow

Creates windows of 1 minutes of data and a label (list of any values available
in that time window):
    x_dm - timestamp, device motion (attitude, rotation rate, user acc,
        gravity, heading)
    x_acc - timestamp, raw accelerometer
    x_loc - timestamp, GPS/location information

Ignored for now....
    y - label for data (if available)

Ignores battery and skips magnetometer (not available on the watch).
"""
import os
import collections

from absl import app
from absl import flags

from files import get_watch_files
from features import parse_full_data
from pool import run_job_pool
from writers import TFRecordWriterFullData, TFRecordWriterFullData2
from full_data_iterator import FullDataIterator
from normalization import to_numpy_if_not

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", None, "Directory where the watch files are (can be in subdirs of this)")
flags.DEFINE_integer("jobs", 0, "How many jobs to run simultaneously (0 = number of cores)")
flags.DEFINE_string("nums", None, "Comma-separated list of which watch numbers to process (e.g. 1,2,3,4,...,15)")
flags.DEFINE_integer("order_window_size", 50, "Size of window to use to make sure the file's out-of-order samples are sorted, done per-sensor")
flags.DEFINE_integer("time_window_size", 60, "Seconds of sensor data for each window")
flags.DEFINE_integer("min_samples_per_window", 100, "Minimum (device motion) samples in a window, otherwise its discarded (e.g. when on charger)")
flags.DEFINE_integer("downsample", 0, "Take every x'th accelerometer and device motion sample (e.g. if 50 Hz data and this is 5, gives 10 Hz) (0 = disabled)")
flags.DEFINE_boolean("split", False, "Split and normalize now rather than in process_full2.py")
flags.DEFINE_boolean("debug", False, "Print debug information")

flags.mark_flag_as_required("dir")
flags.mark_flag_as_required("nums")


def process_watch(watch_number):
    """
    Process the data from one watch from any week (matches multiple subdirs)
    """
    # Get the data files and response files for this watch number
    data_files = get_watch_files(FLAGS.dir, watch_number)
    response_files = get_watch_files(FLAGS.dir, watch_number, responses=True)

    # Disable downsampling if 0
    downsample = FLAGS.downsample

    if downsample == 0:
        downsample = None

    # Iterator for non-overlapping windows of data of time_window_size seconds
    fullDataIter = FullDataIterator(data_files, response_files,
        order_window_size=FLAGS.order_window_size,
        time_window_size=FLAGS.time_window_size,
        downsample=downsample)

    # If desired, split and normalize now rather than doing it in two steps
    # Note: if doing this, requires more memory (e.g. maybe downsample first)
    if FLAGS.split:
        writer = TFRecordWriterFullData2(watch_number)
    else:
        writer = TFRecordWriterFullData(watch_number)

    # Keep track of how many of each category/type we found
    location_categories = collections.defaultdict(int)

    # Save data, if --split
    x_dms = []
    x_accs = []
    x_locs = []

    for window in fullDataIter:
        # Only save window if enough samples in it
        if len(window.dm) > FLAGS.min_samples_per_window:
            x_dm, x_acc, x_loc = parse_full_data(window, location_categories)

            x_dm = to_numpy_if_not(x_dm)
            x_acc = to_numpy_if_not(x_acc)
            x_loc = to_numpy_if_not(x_loc)

            if FLAGS.debug:
                print("x shapes:", x_dm.shape, x_acc.shape, x_loc.shape)

            if FLAGS.split:
                x_dms.append(x_dm)
                x_accs.append(x_acc)
                x_locs.append(x_loc)
            else:
                writer.write_window(x_dm, x_acc, x_loc)

    if FLAGS.split:
        writer.write_records(x_dms, x_accs, x_locs)

    # Debugging
    print("Watch%03d"%watch_number + ": location categories:",
        location_categories)

    writer.close()


def main(argv):
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
