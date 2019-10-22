"""
Writers
"""
import numpy as np

from sklearn.model_selection import train_test_split

from tfrecord import write_tfrecord, tfrecord_filename
from normalization import calc_normalization, apply_normalization


class WriterBase:
    def __init__(self, watch_number):
        self.watch_number = watch_number
        self.filename_prefix = "watch%03d"%watch_number

        # For the summer experiment, TODO put this somewhere else
        self.class_labels = [
            "Cook", "Eat", "Hygiene", "Work", "Exercise", "Travel", "Other",
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

        return np.vstack(windows_x)

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

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


class TFRecordWriter(WriterBase):
    def __init__(self, watch_number,
            window_size=128, window_overlap=False, normalization="meanstd"):
        super().__init__(watch_number)
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

            # Convert label to integer
            y = self.label_to_int(y)

            # Split into windows
            x_windows = self.create_windows_x(x, self.window_size, self.window_overlap)

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

        # Calculate normalization only on the training data
        if self.normalization != "none":
            normalization = calc_normalization(train_data, self.normalization)

            # Apply the normalization to the training, validation, and testing data
            train_data = apply_normalization(train_data, normalization)
            valid_data = apply_normalization(valid_data, normalization)
            test_data = apply_normalization(test_data, normalization)

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
