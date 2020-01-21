"""
Full data iterator
"""
from data_iterator import DataPeekIterator
from watch_data_pb2 import SensorData


class FullData:
    def __init__(self, resp=None, dm=None, acc=None, loc=None):
        self.resp = resp
        self.dm = dm
        self.acc = acc
        self.loc = loc


class FullDataIterator:
    """
    Full data iterator

    Go through all the data getting data from each data source within the
    windows (e.g. all data in each non-overlapping 5 minute window).

    Downsamples accelerometer and device motion data -- take every x'th sample.
    For example, if sampling at 50 Hz, and downsample=5, then it takes every 5th
    sample to get ~10Hz data.

    Usage:
        fullDataIter = FullDataIterator(data_files, response_files,
            order_window_size=50, time_window_size=300, downsample=5)

        for window in fullDataIter:
            windows.append(window)
    """
    def __init__(self, data_files, response_files, order_window_size,
            time_window_size, downsample=None):
        self.respPeekIter = DataPeekIterator(response_files,
            responses=True, window=order_window_size)
        self.dmPeekIter = DataPeekIterator(data_files,
            data_type=SensorData.MESSAGE_TYPE_DEVICE_MOTION,
            window=order_window_size, downsample=downsample)
        self.accPeekIter = DataPeekIterator(data_files,
            data_type=SensorData.MESSAGE_TYPE_ACCELEROMETER,
            window=order_window_size, downsample=downsample)
        self.locPeekIter = DataPeekIterator(data_files,
            data_type=SensorData.MESSAGE_TYPE_LOCATION,
            window=order_window_size)
        self.time_window_size = time_window_size  # seconds
        self.start_epoch = self.initial_start_epoch()  # epoch, not datetime
        self.window_data = None

    def __iter__(self):
        return self

    def get_if_not_end(self, peek_iter):
        """ Get value if not StopIteration, else return None """
        # Try getting next data
        try:
            return next(peek_iter)
        except StopIteration:
            pass

        return None

    def initial_start_epoch(self):
        """ Get the minimum epoch of all the iterators """
        objs = [self.respPeekIter, self.dmPeekIter, self.accPeekIter,
            self.locPeekIter]
        iters = [iter(x) for x in objs]
        firsts = [self.get_if_not_end(x) for x in iters]
        epochs = [x.epoch for x in firsts if x is not None]

        if len(epochs) > 0:
            return min(epochs)
        else:
            return None

    def advance_till_epoch(self, peek_iter, start_epoch, end_epoch):
        """ Advance peek_iter while its epoch < the specified epoch. Save the
        values >= start_epoch and < end_epoch """
        values = []

        for v in peek_iter:
            if v.epoch < end_epoch:
                if v.epoch >= start_epoch:
                    values.append(v)
                peek_iter.pop()
            else:
                break

        return values

    def is_end_of_iterator(self, peek_iter):
        """ Checks if we've reached the end of the iterator

        Note: this only works with peek iterators since we can call next()
        without pop() so that we don't affect the next next() call. """
        is_end = None

        # Try getting next data
        try:
            _ = next(peek_iter)
            is_end = False
        except StopIteration:
            is_end = True

        return is_end

    def next_window(self):
        """ Get the next window of data starting from start_epoch to
        end_epoch = start_epoch + time_window_size (not including end_epoch).
        Advances start_epoch so the next call gets the next window, starting at
        end_epoch.

        If no more data, sets window_data to None. """
        # Check that we're not done already
        #
        # We do this rather than len(resps), ... later on since we may have
        # windows with no data in it (e.g. at night) but still have more data
        # later on.
        finished = self.is_end_of_iterator(self.respPeekIter) \
            and self.is_end_of_iterator(self.dmPeekIter) \
            and self.is_end_of_iterator(self.accPeekIter) \
            and self.is_end_of_iterator(self.locPeekIter)

        if finished:
            self.window_data = None
            return

        # If not done, then get start/end timestamps
        #
        # Note: these are "epochs", i.e. seconds, not actual Python datetime
        # objects
        start_epoch = self.start_epoch
        end_epoch = self.start_epoch + self.time_window_size

        # Get sensor values
        resps = self.advance_till_epoch(self.respPeekIter, start_epoch, end_epoch)
        dms = self.advance_till_epoch(self.dmPeekIter, start_epoch, end_epoch)
        accs = self.advance_till_epoch(self.accPeekIter, start_epoch, end_epoch)
        locs = self.advance_till_epoch(self.locPeekIter, start_epoch, end_epoch)

        # Put lists into an object
        self.window_data = FullData(resps, dms, accs, locs)

        # Update start epoch for the next window
        self.start_epoch = end_epoch

    def __next__(self):
        """ Generate next window, return it if we're not finished with all the
        data """
        self.next_window()

        if self.window_data is not None:
            return self.window_data

        raise StopIteration
