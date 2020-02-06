#!/usr/bin/env python3
"""
Convert the .pb data/response files to .tfrecord files for use in TensorFlow

Creates state vectors at each device motion timestamp of the last sensor
values for all the other sensors (raw accel, location). Ignores battery
and skips magnetometer (not available on the watch).
"""
import os
import collections

from absl import app
from absl import flags
from datetime import datetime, timedelta

from files import get_watch_files
from features import parse_state_vector, parse_response_vector
from pool import run_job_pool
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
    data_files = get_watch_files(FLAGS.dir, watch_number)
    response_files = get_watch_files(FLAGS.dir, watch_number, responses=True)

    # We'll create a state vector for the appropriate time windows nearby the
    # labeled responses, so create iterator over responses and then in the
    # state vector iterator, we'll create the iterators over the data.
    respIter = DataIterator(response_files, responses=True,
        window=FLAGS.window_size)
    statePeekIter = StateVectorPeekIterator(data_files, window=FLAGS.window_size)

    # Generate window for each response
    windows = []
    printed_feature_count = False

    # Keep track of how many of each category/type we found
    location_categories = collections.defaultdict(int)

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
                x.append(parse_state_vector(epoch, dm, acc, loc,
                    location_categories))
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
        y = parse_response_vector(resp, include_other=False)

        if len(x) > 0:
            if FLAGS.debug:
                print("Watch%03d"%watch_number + ": found data for label",
                    y, "at time", str(datetime.fromtimestamp(resp.epoch)),
                    "having", len(x), "time steps")
            windows.append((x, y))
        elif FLAGS.debug:
            print("Watch%03d"%watch_number + ": Warning: no data for label",
                y, "at time", str(datetime.fromtimestamp(resp.epoch)))

    # Debugging
    print("Watch%03d"%watch_number + ": location categories:",
        location_categories)

    # Output
    if FLAGS.output == "tfrecord":
        writer = TFRecordWriter(watch_number)
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
