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

    def create_tf_example_full(self, x_dm, x_acc, x_loc, resp, x_dm_epochs,
            x_acc_epochs, x_loc_epochs, resp_epochs):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'x_dm': _bytes_feature(tf.io.serialize_tensor(x_dm)),
            'x_acc': _bytes_feature(tf.io.serialize_tensor(x_acc)),
            'x_loc': _bytes_feature(tf.io.serialize_tensor(x_loc)),
            'resp': _bytes_feature(tf.io.serialize_tensor(resp)),

            'x_dm_epochs': _bytes_feature(tf.io.serialize_tensor(x_dm_epochs)),
            'x_acc_epochs': _bytes_feature(tf.io.serialize_tensor(x_acc_epochs)),
            'x_loc_epochs': _bytes_feature(tf.io.serialize_tensor(x_loc_epochs)),
            'resp_epochs': _bytes_feature(tf.io.serialize_tensor(resp_epochs)),
        }))
        return tf_example

    def write(self, x_dm, x_acc, x_loc, resp, x_dm_epochs, x_acc_epochs,
            x_loc_epochs, resp_epochs):
        tf_example = self.create_tf_example_full(x_dm, x_acc, x_loc, resp,
            x_dm_epochs, x_acc_epochs, x_loc_epochs, resp_epochs)
        self.writer.write(tf_example.SerializeToString())

    def close(self):
        # Normally the tf.io.TFRecordWriter is used in a with block
        if self.need_to_close:
            self.writer.close()


def tfrecord_filename(dataset_name, postfix):
    """
    Version of tfrecord_filename ignoring the pairs and just creating a
    separate file for each domain. This works if there's no changes in the
    data based on the pairing (e.g. no resizing to match image dimensions)
    """
    return "%s_%s.tfrecord"%(dataset_name, postfix)
