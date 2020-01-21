#!/usr/bin/env python3
"""
Second step of full data processing -- normalize and split into train/valid/test

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.
"""
import os
import numpy as np

from absl import app
from absl import flags

from pool import run_job_pool
from tfrecord import load_tfrecords, tfrecord_filename_full
from writers import TFRecordWriterFullData2

FLAGS = flags.FLAGS

flags.DEFINE_integer("jobs", 0, "How many jobs to run simultaneously (0 = number of cores)")
flags.DEFINE_string("nums", None, "Comma-separated list of which watch numbers to process (e.g. 1,2,3,4,...,15)")
flags.DEFINE_boolean("debug", False, "Print debug information")

flags.mark_flag_as_required("nums")


def get_xs(dataset):
    """ Get all x values in the dataset, convert to numpy arrays """
    x_dms = []
    x_accs = []
    x_locs = []

    for x_dm, x_acc, x_loc in dataset:
        x_dms.append(x_dm.numpy())
        x_accs.append(x_acc.numpy())
        x_locs.append(x_loc.numpy())

        # print(x_dms[-1].shape, x_accs[-1].shape, x_locs[-1].shape)

    # x_dms = np.vstack(x_dms)
    # x_accs = np.vstack(x_accs)
    # x_locs = np.vstack(x_locs)

    return x_dms, x_accs, x_locs


def process_watch(watch_number):
    """ Process the data from one watch """
    # Writer that internally normalizes and splits into train/valid/test
    writer = TFRecordWriterFullData2(watch_number)

    # Filename of raw files from process_full.py
    #
    # Get filename_prefix from writer, since inherits from the same
    # writer used to create the files in process_full.py
    filename = tfrecord_filename_full(writer.filename_prefix)

    # Get data
    dataset = load_tfrecords([filename])
    x_dms, x_accs, x_locs = get_xs(dataset)

    # Write data
    writer.write_records(x_dms, x_accs, x_locs)
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
