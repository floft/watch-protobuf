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
flags.DEFINE_integer("min_samples_per_window", 128, "Minimum (device motion) samples in a window, otherwise its discarded (e.g. when on charger)")
flags.DEFINE_integer("downsample", 10, "Take every x'th accelerometer and device motion sample (e.g. if 50 Hz data and this is 5, gives 10 Hz) (0 = disabled)")
flags.DEFINE_boolean("split", True, "Split and normalize now rather than in process_full2.py")

# Note: be sure to change these if you change time_window_size and downsample above
flags.DEFINE_integer("max_dm_length", 128, "If split -- max device motion time series length (if less than true max, time series is truncated; 0 = max length of data)")
flags.DEFINE_integer("max_acc_length", 128, "If split -- max accelerometer time series length (if less than true max, time series is truncated; 0 = max length of data)")
flags.DEFINE_integer("max_loc_length", 1, "If split -- max location time series length (if less than true max, time series is truncated; 0 = max length of data)")
flags.DEFINE_integer("max_resp_length", 1, "If split -- max labels time series length (if less than true max, time series is truncated; 0 = max length of data)")

flags.DEFINE_boolean("debug", False, "Print debug information (use to inform setting --max_{dm,acc,loc}_length=...)")

flags.mark_flag_as_required("dir")
flags.mark_flag_as_required("nums")


def is_labeled(resp):
    """
    Determine if there's a non-zero (i.e. not "Unknown") label. Note: the actual
    unknown labels will come from "Other" since we map that to unknown. The
    other unknowns will come from the pad-to-length-one part when writing to the
    tfrecord file. It's padded with zero, which we map to "Unknown" as well.

    When parsing the files, only classify when it's not 0. Set softmax size to
    exclude the "Unknown" label (since we never want to predict it).

    resp = [np.array(label1), np.array(label2), ...]
    """
    for r in resp:
        if r != 0:
            return True

    return False


def process_watch(watch_number):
    """
    Process the data from one watch from any week (matches multiple subdirs)
    """
    # TODO implement writing to separate labeled/unlabeled files
    assert FLAGS.split, "split=False is not implemented"

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
        labeled_writer = TFRecordWriterFullData2(watch_number,
            prefix="watch_raw_labeled")
        unlabeled_writer = TFRecordWriterFullData2(watch_number,
            prefix="watch_raw_unlabeled")
    else:
        writer = TFRecordWriterFullData(watch_number)

    # Keep track of how many of each category/type we found
    location_categories = collections.defaultdict(int)

    # Save data, if --split
    unlabeled_x_dms = []
    unlabeled_x_accs = []
    unlabeled_x_locs = []
    unlabeled_resps = []
    unlabeled_x_dms_epochs = []
    unlabeled_x_accs_epochs = []
    unlabeled_x_locs_epochs = []
    unlabeled_resps_epochs = []

    labeled_x_dms = []
    labeled_x_accs = []
    labeled_x_locs = []
    labeled_resps = []
    labeled_x_dms_epochs = []
    labeled_x_accs_epochs = []
    labeled_x_locs_epochs = []
    labeled_resps_epochs = []

    for window in fullDataIter:
        # Only save window if enough samples in it
        if len(window.dm) >= FLAGS.min_samples_per_window:
            x_dm, x_acc, x_loc, resp, \
                x_dm_epochs, x_acc_epochs, x_loc_epochs, resp_epochs \
                = parse_full_data(window, location_categories)

            x_dm = to_numpy_if_not(x_dm)
            x_acc = to_numpy_if_not(x_acc)
            x_loc = to_numpy_if_not(x_loc)
            resp = to_numpy_if_not(resp)
            x_dm_epochs = to_numpy_if_not(x_dm_epochs)
            x_acc_epochs = to_numpy_if_not(x_acc_epochs)
            x_loc_epochs = to_numpy_if_not(x_loc_epochs)
            resp_epochs = to_numpy_if_not(resp_epochs)

            if FLAGS.debug:
                print("x shapes:", x_dm.shape, x_acc.shape, x_loc.shape,
                    x_dm_epochs.shape, x_acc_epochs.shape, x_loc_epochs.shape)
                print("y shapes:", resp.shape, resp_epochs.shape)

            if FLAGS.split:
                if is_labeled(resp):
                    labeled_x_dms.append(x_dm)
                    labeled_x_accs.append(x_acc)
                    labeled_x_locs.append(x_loc)
                    labeled_resps.append(resp)
                    labeled_x_dms_epochs.append(x_dm_epochs)
                    labeled_x_accs_epochs.append(x_acc_epochs)
                    labeled_x_locs_epochs.append(x_loc_epochs)
                    labeled_resps_epochs.append(resp_epochs)
                else:
                    unlabeled_x_dms.append(x_dm)
                    unlabeled_x_accs.append(x_acc)
                    unlabeled_x_locs.append(x_loc)
                    unlabeled_resps.append(resp)
                    unlabeled_x_dms_epochs.append(x_dm_epochs)
                    unlabeled_x_accs_epochs.append(x_acc_epochs)
                    unlabeled_x_locs_epochs.append(x_loc_epochs)
                    unlabeled_resps_epochs.append(resp_epochs)
            else:
                writer.write_window(x_dm, x_acc, x_loc, resp,
                    x_dm_epochs, x_acc_epochs, x_loc_epochs, resp_epochs)

    if FLAGS.split:
        labeled_writer.write_records(labeled_x_dms, labeled_x_accs,
            labeled_x_locs, labeled_resps, labeled_x_dms_epochs,
            labeled_x_accs_epochs, labeled_x_locs_epochs, labeled_resps_epochs)
        unlabeled_writer.write_records(unlabeled_x_dms, unlabeled_x_accs,
            unlabeled_x_locs, unlabeled_resps, unlabeled_x_dms_epochs,
            unlabeled_x_accs_epochs, unlabeled_x_locs_epochs,
            unlabeled_resps_epochs)

    # Debugging
    print("Watch%03d"%watch_number + ": location categories:",
        location_categories)

    labeled_writer.close()
    unlabeled_writer.close()


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
