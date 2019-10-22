"""
Writers
"""
import numpy as np
import tensorflow as tf


class WriterBase:
    def __init__(self, watch_number):
        self.watch_number = watch_number
        self.filename_prefix = "watch%03d"%watch_number


class TFRecordWriter(WriterBase):
    def __init__(self, watch_number, num_time_steps):
        super().__init__(watch_number)
        self.num_time_steps = num_time_steps

    def pad_to(self, data, desired_length):
        """ Pads time steps to the desired length:
        Before: (time_steps, num_features)
        After: (desired_length, num_features)
        """
        current_length = data.shape[0]
        assert current_length <= desired_length, "Cannot shrink size by padding"
        return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                mode="constant", constant_values=0)

    def process_records(self, records):
        new_records = []

        for x, y in records:
            # Create a numpy array and convert None values to 0
            x = np.array(x, dtype=np.float32)
            x[np.isnan(x)] = 0.0

            # When processing in batches, we require that each example in a
            # batch have the same number of time steps.
            x = self.pad_to(x, self.num_time_steps)

            new_records.append(x, y)

        return new_records

    def write_records(self, records):
        records = self.process_records(records)

        # TODO normalization, etc. see generate_tfrecords.py
        # TODO split into smaller windows, maybe 128 samples, i.e. ~11 windows/label
        # TODO write


class CSVWriter(WriterBase):
    """ Within each time step the values are comma separated, but then they
    are semicolon separated. Then there's another ; followed by the label.
    For example: 0,0,...;0,0,...;Walk """
    def write_records(self, records):
        filename = self.filename_prefix+".csv"

        with open(filename, "w") as f:
            for x, y in records:
                print("Values:", x, y)
                print()
                print("Length:", len(x))
                print()
                time_steps = [",".join([str(f) for f in ts]) for ts in x]
                print("Time Steps:", time_steps)
                print()
                print("All x:", ";".join(time_steps))
                print()

                f.write(";".join([",".join([str(f) for f in ts]) for ts in x]) + ";" + y + "\n")


class JSONWriter(WriterBase):
    def write_records(self, records):
        pass
