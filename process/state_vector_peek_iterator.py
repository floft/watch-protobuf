"""
State vector peek iterator
"""
from data_iterator import DataIterator, DataPeekIterator
from watch_data_pb2 import SensorData


class StateVectorPeekIterator:
    """
    State vector peek iterator

    Generate the state vector consisting of the various sensor data.

    Usage:
        statePeekIter = StateVectorPeekIterator(data_files, window=50)

        for state in statePeekIter:
            if some_condition_is_met:  # probably using state["epoch"]
                states.append(state)
                statePeekIter.pop()
            else:
                break
    """
    def __init__(self, data_files, window):
        self.dmIter = DataIterator(data_files,
            data_type=SensorData.MESSAGE_TYPE_DEVICE_MOTION, window=window)
        self.accPeekIter = DataPeekIterator(data_files,
            data_type=SensorData.MESSAGE_TYPE_ACCELEROMETER, window=window)
        self.locPeekIter = DataPeekIterator(data_files,
            data_type=SensorData.MESSAGE_TYPE_LOCATION, window=window)
        self.state_vector = None
        self.last_acc = None
        self.last_loc = None

    def __iter__(self):
        return self

    def advance_till_epoch(self, peek_iter, epoch):
        """ Advance the peek_iter while its epoch < the specified epoch,
        i.e. find/return the value with the max epoch <= to the specified
        epoch """
        last = None

        for v in peek_iter:
            if v["epoch"] < epoch:
                last = v
                peek_iter.pop()
            else:
                break

        return last

    def next_state_vector(self):
        """ Create next state vector -- align other sensors to the device
        motion data. If no more data, set the state vector to None. """
        found = False

        # Try getting next DM data
        try:
            dm = next(self.dmIter)
            found = True
        except StopIteration:
            found = False

        if found:
            # Get other sensor values closest but <= to DM's epoch
            last_acc = self.advance_till_epoch(self.accPeekIter, dm["epoch"])
            last_loc = self.advance_till_epoch(self.locPeekIter, dm["epoch"])

            # We might have already gotten all the acc/loc data from the files,
            # so keep using the previous last values if we don't have any more.
            if last_acc is not None:
                self.last_acc = last_acc
            if last_loc is not None:
                self.last_loc = last_loc

            # Create the state vector from all the different data parts
            self.state_vector = self.create_state_vector(dm, self.last_acc,
                self.last_loc)
        else:
            self.state_vector = None

    def pop(self):
        """
        Pop off and return the last state vector

        Note: we actually create the next one here too.
        """
        state_vector = self.state_vector
        self.next_state_vector()
        return state_vector

    def __next__(self):
        """
        Return the next state vector

        Actually, we don't create it at this point since we support peeking.
        Just return the last one created. During pop() we'll create the next one.

        Though, if it's the first iteration, we have to create the state vector
        here.
        """
        if self.state_vector is None:
            self.next_state_vector()

        if self.state_vector is not None:
            return self.state_vector

        raise StopIteration

    def create_state_vector(self, dm, acc, loc):
        # Note: values near the beginning of the file may be None since there
        # may not have been data from these sensors before then.
        return {
            "epoch": dm["epoch"],
            "data": [
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
        }
