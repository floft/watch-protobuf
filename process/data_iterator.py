"""
Data iterators
"""
import heapq

from decoding import decode
from watch_data_pb2 import SensorData, PromptResponse


class SortOnEpoch:
    """ The heap needs to be sorted on the epoch, so override the < operator
    to sort on the passed in data's "epoch" value. Use .get() to return data. """
    def __init__(self, data):
        self.data = data

    def __lt__(self, other):
        return self.data.epoch < other.data.epoch

    def get(self):
        return self.data


class NeedLargerWindowError(Exception):
    """
    If the new item's epoch is not >= the max of all previous ones, then
    you need to increase the window size
    """
    pass


class DataIteratorBase:
    def __init__(self, filenames, window, data_type=None, responses=False,
            downsample=None):
        self.filenames = filenames
        self.window = window
        assert self.window > 0, "window size must be positive"
        self.data_type = data_type
        self.responses = responses
        self.heap = []
        self.messages = []

        if self.responses:
            self.msg_type = PromptResponse
        else:
            self.msg_type = SensorData

        # Keep track of which file and message we're on (indices into
        # self.filenames and self.messages)
        self.file_index = 0
        self.message_index = 0

        # Keep track of latest epoch so we know if we ever go back in time,
        # i.e. the window isn't large enough
        self.last_epoch = 0

        # Keep track of if we wish to downsample the data
        self.downsample = downsample
        self.downsample_index = 0

    def _fill_window(self):
        # Keep going till we fill the window
        while len(self.heap) < self.window:
            # If we ran out of messages, load from the next file if there is
            # one, otherwise we're out of messages. This is a loop since the
            # next file may be blank, so we may have to do this a number of
            # times.
            while self.message_index >= len(self.messages):
                if self.file_index < len(self.filenames):
                    self.messages = decode(self.filenames[self.file_index], self.msg_type)
                    self.file_index += 1
                    self.message_index = 0
                else:
                    return  # No more files/data

            msg = self.messages[self.message_index]

            # Save if it's a response, but if a data type, then if we want to
            # only keep certain messages, skip if it's not the type we want.
            if self.responses or self.data_type is None \
                    or msg.message_type == self.data_type:

                # And, if it is the data type we care about, then if
                # downsampling, only take every x'th of these samples
                self.downsample_index += 1

                if self.downsample is None or \
                        self.downsample_index%self.downsample == 0:
                    self._push(msg)

            self.message_index += 1

    def _push(self, item):
        heapq.heappush(self.heap, SortOnEpoch(item))

    def pop(self):
        if len(self.heap) > 0:
            # Note: data on heap is SortOnEpoch objects, so get .data for the
            # original data
            item = heapq.heappop(self.heap).data

            # Verify we're sorted -- i.e. that a new element is never older than
            # the previously returned maximum epoch
            epoch = item.epoch
            if epoch < self.last_epoch:
                raise NeedLargerWindowError
            self.last_epoch = max(self.last_epoch, epoch)

            return item
        return None

    def peek(self):
        if len(self.heap) > 0:
            return self.heap[0].data
        return None

    def __iter__(self):
        return self

    def __next__(self):
        self._fill_window()
        item = self.pop()

        if item is not None:
            return item

        raise StopIteration


class DataIterator(DataIteratorBase):
    """
    Timestamp/epoch-sorted iteration/parsing over protobuf files

    Provides an iterator over the passed in protobuf filenames that sorts the
    data as long as unsorted values are within the window (errors if not).

    Note: filenames are not sorted here, so probably sort by timestamp before
    passing to this class.
    """
    pass


class DataPeekIterator(DataIteratorBase):
    """
    Timestamp/epoch-sorted iteration/parsing over protobuf files (peek version)

    Provides a "peek" version that does not consume/pop the value returned by
    the iterator until iter.pop() is called. This is useful when you need to
    peek at the next value but if it doesn't meet some condition, then you'll
    break the loop and look at it again next iteration.
    """
    def __next__(self):
        self._fill_window()
        item = self.peek()

        if item is not None:
            return item

        raise StopIteration
