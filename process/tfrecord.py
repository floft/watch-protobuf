"""
Functions to write the x,y data to a tfrecord file
"""
import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(x, y):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
    }))
    return tf_example


def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i])
            writer.write(tf_example.SerializeToString())


class FullTFRecord:
    def __init__(self, filename, experimental=False):
        if not experimental:
            self.need_to_close = True
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            self.writer = tf.io.TFRecordWriter(filename, options=options)
        else:
            # Errors on write() due to:
            # `dataset` must be a `tf.data.Dataset` object ???
            #
            # Maybe you can't write individual examples but have to write the
            # whole dataset at once?
            self.need_to_close = False
            self.writer = tf.data.experimental.TFRecordWriter(filename,
                compression_type="GZIP")

    def create_tf_example_full(self, x_dm, x_acc, x_loc, x_dm_epochs,
            x_acc_epochs, x_loc_epochs):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'x_dm': _bytes_feature(tf.io.serialize_tensor(x_dm)),
            'x_acc': _bytes_feature(tf.io.serialize_tensor(x_acc)),
            'x_loc': _bytes_feature(tf.io.serialize_tensor(x_loc)),

            'x_dm_epochs': _bytes_feature(tf.io.serialize_tensor(x_dm_epochs)),
            'x_acc_epochs': _bytes_feature(tf.io.serialize_tensor(x_acc_epochs)),
            'x_loc_epochs': _bytes_feature(tf.io.serialize_tensor(x_loc_epochs)),
        }))
        return tf_example

    def write(self, x_dm, x_acc, x_loc, x_dm_epochs, x_acc_epochs, x_loc_epochs):
        tf_example = self.create_tf_example_full(x_dm, x_acc, x_loc,
            x_dm_epochs, x_acc_epochs, x_loc_epochs)
        self.writer.write(tf_example.SerializeToString())

    def close(self):
        # Normally the tf.io.TFRecordWriter is used in a with block
        if self.need_to_close:
            self.writer.close()


def tfrecord_filename(dataset_name, train_or_test, raw=False):
    """
    Version of tfrecord_filename ignoring the pairs and just creating a
    separate file for each domain. This works if there's no changes in the
    data based on the pairing (e.g. no resizing to match image dimensions)
    """
    # Sanity checks
    assert train_or_test in ["train", "valid", "test"], \
        "train_or_test must be train, valid, or test"

    if raw:
        filename = "%s_raw_%s.tfrecord"%(dataset_name, train_or_test)
    else:
        filename = "%s_%s.tfrecord"%(dataset_name, train_or_test)

    return filename


def tfrecord_filename_full(prefix):
    """ Filename for raw data """
    return prefix + "_raw.tfrecord"


def load_tfrecords(filenames, batch_size=None):
    """ Load data from .tfrecord files, for details see corresponding function
    in CoDATS load_datasets.py """
    if len(filenames) == 0:
        return None

    feature_description = {
        'x_dm': tf.io.FixedLenFeature([], tf.string),
        'x_acc': tf.io.FixedLenFeature([], tf.string),
        'x_loc': tf.io.FixedLenFeature([], tf.string),

        'x_dm_epochs': tf.io.FixedLenFeature([], tf.string),
        'x_acc_epochs': tf.io.FixedLenFeature([], tf.string),
        'x_loc_epochs': tf.io.FixedLenFeature([], tf.string),

        # 'y': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_example_function(example_proto):
        parsed = tf.io.parse_single_example(serialized=example_proto,
            features=feature_description)
        x_dm = tf.io.parse_tensor(parsed["x_dm"], tf.float32)
        x_acc = tf.io.parse_tensor(parsed["x_acc"], tf.float32)
        x_loc = tf.io.parse_tensor(parsed["x_loc"], tf.float32)

        x_dm_epochs = tf.io.parse_tensor(parsed["x_dm_epochs"], tf.float32)
        x_acc_epochs = tf.io.parse_tensor(parsed["x_acc_epochs"], tf.float32)
        x_loc_epochs = tf.io.parse_tensor(parsed["x_loc_epochs"], tf.float32)

        # y = tf.io.parse_tensor(parsed["y"], tf.float32)
        return x_dm, x_acc, x_loc, x_dm_epochs, x_acc_epochs, x_loc_epochs

    files = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP').prefetch(100),
        cycle_length=len(filenames), block_length=1)
    dataset = dataset.map(_parse_example_function)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)

    return dataset
