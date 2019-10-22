#!/usr/bin/env python3
"""
Convert the .pb data/response files to .tfrecord files for use in TensorFlow

Creates state vectors at each device motion timestamp of the last sensor
values for all the other sensors (raw accel, location). Ignores battery
and skips magnetometer (not available on the watch).
"""
import os
import pathlib

from absl import app
from absl import flags
from datetime import datetime, timedelta

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


def parse_state_vector(epoch, dm, acc, loc):
    """ Parse here rather than in DataIterator since we end up skipping lots
    of data, so if we do it now, we'll run the parsing way fewer times """
    if dm is not None:
        dm = parse_data(dm)
    if acc is not None:
        acc = parse_data(acc)
    if loc is not None:
        loc = parse_data(loc)

    return [
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
            else:
                # Don't consume - we'll look at this state for the next
                # window
                break

        # Save this window if we have data for it
        if len(x) > 0:
            y = parse_response(resp)["label"]
            windows.append((x, y))

    # Output
    if FLAGS.output == "tfrecord":
        writer = TFRecordWriter(watch_number)
    elif FLAGS.output == "csv":
        writer = CSVWriter(watch_number)
    else:
        raise NotImplementedError("unsupported output format: "+FLAGS.output)

    writer.write_records(windows)


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
