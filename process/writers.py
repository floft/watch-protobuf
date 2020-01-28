"""
Writers
"""
import numpy as np
import tensorflow as tf

from absl import flags
from sklearn.model_selection import train_test_split

from tfrecord import write_tfrecord, tfrecord_filename, \
    tfrecord_filename_full, FullTFRecord
from normalization import calc_normalization, calc_normalization_jagged, \
    apply_normalization, apply_normalization_jagged, to_numpy_if_not


FLAGS = flags.FLAGS


class WriterBase:
    def __init__(self, watch_number, include_other=True):
        self.watch_number = watch_number
        self.include_other = include_other

        # For the summer experiment, TODO put this somewhere else
        if self.include_other:
            self.filename_prefix = "watch_"+str(watch_number)
            self.class_labels = [
                "Cook", "Eat", "Hygiene", "Work", "Exercise", "Travel", "Other",
            ]
            self.num_classes = len(self.class_labels)
        else:
            self.filename_prefix = "watch_noother_"+str(watch_number)
            self.class_labels = [
                "Cook", "Eat", "Hygiene", "Work", "Exercise", "Travel",
            ]
            self.num_classes = len(self.class_labels)

    def train_test_split(self, x, y, test_percent=0.2, random_state=42):
        """
        Split x and y data into train/test sets

        Warning: train_test_split() is from sklearn but self.train_test_split()
        is this function, which is what you should use.
        """
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=test_percent,
            stratify=y, random_state=random_state)
        return x_train, y_train, x_test, y_test

    def train_test_split_xs(self, x_dm, x_acc, x_loc, test_percent=0.2,
            random_state=42):
        x_dm_train, x_dm_test, x_acc_train, x_acc_test, x_loc_train, x_loc_test = \
            train_test_split(x_dm, x_acc, x_loc, test_size=test_percent,
                random_state=random_state)
        return x_dm_train, x_acc_train, x_loc_train, \
            x_dm_test, x_acc_test, x_loc_test

    def valid_split(self, data, labels, seed=None, validation_size=1000):
        """ (Stratified) split training data into train/valid as is commonly done,
        taking 1000 random (stratified) (labeled, even if target domain) samples for
        a validation set """
        percentage_size = int(0.2*len(data))
        if percentage_size > validation_size:
            test_size = validation_size
        else:
            print("Warning: using smaller validation set size", percentage_size)
            test_size = 0.2  # 20% maximum

        x_train, x_valid, y_train, y_valid = \
            train_test_split(data, labels, test_size=test_size,
                stratify=labels, random_state=seed)

        return x_valid, y_valid, x_train, y_train

    def valid_split_xs(self, x_dm, x_acc, x_loc, seed=None, validation_size=1000):
        """ (Stratified) split training data into train/valid as is commonly done,
        taking 1000 random (stratified) (labeled, even if target domain) samples for
        a validation set """
        assert len(x_dm) == len(x_acc)
        assert len(x_dm) == len(x_loc)

        percentage_size = int(0.2*len(x_dm))
        if percentage_size > validation_size:
            test_size = validation_size
        else:
            print("Warning: using smaller validation set size", percentage_size)
            test_size = 0.2  # 20% maximum

        x_dm_train, x_dm_valid, x_acc_train, x_acc_valid, x_loc_train, x_loc_valid = \
            train_test_split(x_dm, x_acc, x_loc, test_size=test_size,
                random_state=seed)

        return x_dm_train, x_acc_train, x_loc_train, \
            x_dm_valid, x_acc_valid, x_loc_valid

    def create_windows_x(self, x, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process x (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        x = np.expand_dims(x, axis=1)

        # No work required if the window size is 1, only part required is
        # the above expand dims
        if window_size == 1:
            return x

        windows_x = []
        i = 0

        while i < len(x)-window_size:
            window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
            windows_x.append(window_x)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        if len(windows_x) > 0:
            return np.vstack(windows_x)
        else:
            return None

    def pad_to(self, data, desired_length):
        """
        Pad the number of time steps to the desired length

        Accepts data in one of two formats:
            - shape: (time_steps, features) -> (desired_length, features)
            - shape: (batch_size, time_steps, features) ->
                (batch_size, desired_length, features)
        """
        if len(data.shape) == 2:
            current_length = data.shape[0]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        elif len(data.shape) == 3:
            current_length = data.shape[1]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, 0), (0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        else:
            raise NotImplementedError("pad_to requires 2 or 3-dim data")

    def normalize(self, train_data, valid_data, test_data,
            normalization_method, jagged=False):
        # Which functions to use
        calc = calc_normalization
        apply = apply_normalization

        if jagged:
            calc = calc_normalization_jagged
            apply = apply_normalization_jagged

        if normalization_method != "none":
            # Skip if no data
            if len(train_data) > 0:
                # Calculate normalization only on the training data
                normalization = calc(train_data, normalization_method)

                # Apply the normalization to the training, validation, and testing data
                train_data = apply(train_data, normalization)
                valid_data = apply(valid_data, normalization)
                test_data = apply(test_data, normalization)

        return train_data, valid_data, test_data

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


class TFRecordWriter(WriterBase):
    def __init__(self, watch_number,
            window_size=128, window_overlap=False, normalization="meanstd",
            **kwargs):
        super().__init__(watch_number, **kwargs)
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.normalization = normalization

    def split_into_windows(self, records):
        xs = []
        ys = []

        for x, y in records:
            # Create a numpy array and convert None values to 0
            x = np.array(x, dtype=np.float32)
            x[np.isnan(x)] = 0.0

            # If desired, skip the other class
            if not self.include_other and y == "Other":
                continue

            # Convert label to integer
            y = self.label_to_int(y)

            # Split into windows
            x_windows = self.create_windows_x(x, self.window_size, self.window_overlap)

            # Skip if we couldn't split into windows, probably because there's
            # not enough data in this example to have even one window.
            #
            # Guess at cause, either:
            # - Labels too close: first label consumes most of the data that
            #   would be used by the second. Maybe they tapped multiple times,
            #   tried correcting an incorrect label, etc.
            # - Label comes too soon after taking off the charger, when it
            #   doesn't collect data. It takes some time to start/stop
            #   collecting data (only checks every so many seconds).
            #
            # This may yield shorter windows or no time steps at all, depending
            # on the proximity to the prior label or taking off the charger.
            if x_windows is not None:
                # We have the same label for all the x windows that we split x into,
                # so just copy it that many times
                y_windows = [y]*len(x_windows)

                xs.append(x_windows)
                ys.append(y_windows)

        xs = np.vstack(xs).astype(np.float32)
        ys = np.hstack(ys).astype(np.float32)

        print(xs.shape, ys.shape)

        return xs, ys

    def write(self, filename, x, y):
        if x is not None and y is not None:
            write_tfrecord(filename, x, y)
        else:
            print("Skipping:", filename, "(no data)")

    def write_records(self, records):
        train_filename = tfrecord_filename(self.filename_prefix, "train")
        valid_filename = tfrecord_filename(self.filename_prefix, "valid")
        test_filename = tfrecord_filename(self.filename_prefix, "test")

        # Convert to numpy and split into windows
        xs, ys = self.split_into_windows(records)

        # Split into train/test sets
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(xs, ys)

        # Further split training into train/valid sets
        valid_data, valid_labels, train_data, train_labels = \
            self.valid_split(train_data, train_labels, seed=0)

        # Normalize
        train_data, valid_data, test_data = self.normalize(
            train_data, valid_data, test_data, self.normalization)

        # Saving
        self.write(train_filename, train_data, train_labels)
        self.write(valid_filename, valid_data, valid_labels)
        self.write(test_filename, test_data, test_labels)


class CSVWriter(WriterBase):
    """ Within each time step the values are comma separated, but then they
    are semicolon separated. Then there's another ; followed by the label.
    For example: 0,0,...;0,0,...;Walk

    Note: this is basically just for debugging. It doesn't split it into
    windows, train/valid/test sets, etc.
    """
    def write_records(self, records):
        filename = self.filename_prefix+".csv"

        with open(filename, "w") as f:
            for x, y in records:
                # print("Values:", x, y)
                # print()
                # print("Length:", len(x))
                # print()
                # time_steps = [",".join([str(f) for f in ts]) for ts in x]
                # print("Time Steps:", time_steps)
                # print()
                # print("All x:", ";".join(time_steps))
                # print()

                f.write(";".join([",".join([str(f) for f in ts]) for ts in x]) + ";" + y + "\n")


class TFRecordWriterFullData(WriterBase):
    """ Write raw full data -- part 1 """
    def __init__(self, watch_number, **kwargs):
        super().__init__(watch_number, **kwargs)
        filename = tfrecord_filename_full(self.filename_prefix)
        self.record_writer = FullTFRecord(filename)

    def clean(self, x):
        # # Create a numpy array and convert None values to 0
        # x = np.array(x, dtype=np.float32)
        # x[np.isnan(x)] = 0.0

        # We can clean it later if we want... for now just convert to numpy
        # x = np.array(x, dtype=np.float32)

        # return x

        return to_numpy_if_not(x)

    def write_window(self, x_dm, x_acc, x_loc):
        x_dm = self.clean(x_dm)
        x_acc = self.clean(x_acc)
        x_loc = self.clean(x_loc)

        self.record_writer.write(x_dm, x_acc, x_loc)

    def close(self):
        self.record_writer.close()


class TFRecordWriterFullData2(WriterBase):
    """ Write normalized train/valid/test full data -- part 2 """
    def __init__(self, watch_number, normalization="meanstd", **kwargs):
        super().__init__(watch_number, **kwargs)
        self.normalization = normalization

    def clean(self, x):
        x = to_numpy_if_not(x)

        # convert None values to 0
        x[np.isnan(x)] = 0.0

        return x

    def write(self, filename, x_dm, x_acc, x_loc):
        assert len(x_dm) == len(x_acc)
        assert len(x_dm) == len(x_loc)

        record_writer = FullTFRecord(filename)

        for i in range(len(x_dm)):
            # self.record_writer.write(x_dm[i], x_acc[i], x_loc[i])

            # Clean data first
            x_dm_cur = self.clean(x_dm[i])
            x_acc_cur = self.clean(x_acc[i])
            x_loc_cur = self.clean(x_loc[i])

            record_writer.write(x_dm_cur, x_acc_cur, x_loc_cur)

        record_writer.close()

    def pad(self, x, maxlen):
        """
        Pad inputs to be same length in the time dimension so we can batch
        https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
        """
        # Done on all examples at once
        # Shape of x should be something like: [num_examples, time_steps, features]
        return tf.keras.preprocessing.sequence.pad_sequences(
            x, maxlen=maxlen, dtype='float32', padding='post', truncating='pre',
            value=0.0)

    def write_records(self, x_dm, x_acc, x_loc):
        """ Pass in x_dm = [x_dm1, x_dm2, ...] and similarly x_acc and x_loc """
        train_filename = tfrecord_filename(self.filename_prefix, "train", raw=True)
        valid_filename = tfrecord_filename(self.filename_prefix, "valid", raw=True)
        test_filename = tfrecord_filename(self.filename_prefix, "test", raw=True)

        # Split into train/test sets
        x_dm_train, x_acc_train, x_loc_train, x_dm_test, x_acc_test, x_loc_test = \
            self.train_test_split_xs(x_dm, x_acc, x_loc)

        # Further split training into train/valid sets
        x_dm_train, x_acc_train, x_loc_train, x_dm_valid, x_acc_valid, x_loc_valid = \
            self.valid_split_xs(x_dm_train, x_acc_train, x_loc_train, seed=0)

        # Normalize
        x_dm_train, x_dm_valid, x_dm_test = self.normalize(
            x_dm_train, x_dm_valid, x_dm_test, self.normalization, jagged=True)
        x_acc_train, x_acc_valid, x_acc_test = self.normalize(
            x_acc_train, x_acc_valid, x_acc_test, self.normalization, jagged=True)
        x_loc_train, x_loc_valid, x_loc_test = self.normalize(
            x_loc_train, x_loc_valid, x_loc_test, self.normalization, jagged=True)

        # Pad/truncate to the right length -- after normalizing so the padded
        # 0 values don't affect the mean, etc.
        max_dm_length = FLAGS.max_dm_length
        max_acc_length = FLAGS.max_acc_length
        max_loc_length = FLAGS.max_loc_length

        if max_dm_length == 0:
            max_dm_length = None
        if max_acc_length == 0:
            max_acc_length = None
        if max_loc_length == 0:
            max_loc_length = None

        x_dm_train = self.pad(x_dm_train, max_dm_length)
        x_dm_valid = self.pad(x_dm_valid, max_dm_length)
        x_dm_test = self.pad(x_dm_test, max_dm_length)
        x_acc_train = self.pad(x_acc_train, max_acc_length)
        x_acc_valid = self.pad(x_acc_valid, max_acc_length)
        x_acc_test = self.pad(x_acc_test, max_acc_length)
        x_loc_train = self.pad(x_loc_train, max_loc_length)
        x_loc_valid = self.pad(x_loc_valid, max_loc_length)
        x_loc_test = self.pad(x_loc_test, max_loc_length)

        # TODO remove this -- just checking if it's the 300, 300, 1 like set in
        # the FLAGS.max_{dm,acc,loc}_length
        print("Watch", self.watch_number)
        if len(x_dm_train) > 0:
            print("DM train shape:", x_dm_train[0].shape)
        if len(x_acc_train) > 0:
            print("Acc train shape:", x_acc_train[0].shape)
        if len(x_loc_train) > 0:
            print("Loc train shape:", x_loc_train[0].shape)

        # Saving
        self.write(train_filename, x_dm_train, x_acc_train, x_loc_train)
        self.write(valid_filename, x_dm_valid, x_acc_valid, x_loc_valid)
        self.write(test_filename, x_dm_test, x_acc_test, x_loc_test)

    def close(self):
        pass
