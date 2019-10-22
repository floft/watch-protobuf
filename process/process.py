#!/usr/bin/env python3
"""
Convert the .pb data/response files to .tfrecord files for use in TensorFlow

Creates state vectors at each device motion timestamp of the last sensor
values for all the other sensors (raw accel, gyro, location). Ignores battery
and skips magnetometer (not available on the watch).
"""
import os
import pathlib

from absl import app
from absl import flags
from datetime import datetime, timedelta

from pool import run_job_pool
from data_iterator import DataIterator
from writers import TFRecordWriter, CSVWriter, JSONWriter
from state_vector_peek_iterator import StateVectorPeekIterator

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", None, "Directory where the watch files are (can be in subdirs of this)")
flags.DEFINE_integer("jobs", 0, "How many jobs to run simultaneously (0 = number of cores)")
flags.DEFINE_string("nums", None, "Comma-separated list of which watch numbers to process (e.g. 1,2,3,4,...,15)")
flags.DEFINE_integer("window_size", 5, "Size of window to use to make sure the file's out-of-order samples are sorted, done per-sensor")
flags.DEFINE_integer("begin_offset", -60, "Seconds offset to start looking at data after (before is negative) the label timestamp")
flags.DEFINE_integer("end_offset", -30, "Seconds offset to end looking at data after (before is negative) the label timestamp")
flags.DEFINE_integer("hz", 50, "Frequency data was recorded at, used to pad time steps to hz*(end_offset-begin_offset) -- only used with --output=tfrecord")
flags.DEFINE_boolean("split", True, "Whether to split into train/valid/test data files")
flags.DEFINE_enum("output", "tfrecord", ["tfrecord", "csv", "json"], "Output format to save windows using")

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


def process_watch(watch_number):
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
        beginTime = datetime.fromtimestamp(resp["epoch"]) \
            + timedelta(seconds=FLAGS.begin_offset)
        endTime = datetime.fromtimestamp(resp["epoch"]) \
            + timedelta(seconds=FLAGS.end_offset)

        # Skip state vectors up till the begin time, then stop when we
        # reach end time. Note: this is a "peek" iterator -- if the state's
        # value is beyond the endTime and will be potentially used in the next
        # window, then don't consume/pop it. Otherwise, though, pop the
        # state so that the next iterator call will be different (not inf loop).
        x = []

        for state in statePeekIter:
            epoch = datetime.fromtimestamp(state["epoch"])

            if epoch < beginTime:
                statePeekIter.pop()
                continue
            elif epoch <= endTime:
                x.append(state["data"])
                statePeekIter.pop()
            else:
                # Don't consume - we'll look at this state for the next
                # window
                break

        # Save this window if we have data for it
        if len(x) > 0:
            y = resp["label"]
            windows.append((x, y))

    # TODO
    if FLAGS.split:
        pass

    # Output
    if FLAGS.output == "tfrecord":
        # TODO this will error if there's any too large, check for that
        max_num_time_steps = FLAGS.hz*(FLAGS.end_offset - FLAGS.begin_offset)
        writer = TFRecordWriter(watch_number, max_num_time_steps)
    elif FLAGS.output == "csv":
        writer = CSVWriter(watch_number)
    elif FLAGS.output == "json":
        writer = JSONWriter(watch_number)
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
