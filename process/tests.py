#!/usr/bin/env python3
"""
Test various parts of the processing
"""
import os
import unittest
import numpy as np

from data_iterator import DataIterator, NeedLargerWindowError
from state_vector_peek_iterator import StateVectorPeekIterator
from watch_data_pb2 import SensorData


class ProtobufTestsBase(unittest.TestCase):
    """ Base class that defines the paths to data files for the tests """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = "test_data/"

        self.response_files_wrong_order = [
            os.path.join(path, "responses_20190613_170934.pb"),
            os.path.join(path, "responses_20190613_170415.pb"),
        ]
        self.response_files = self.response_files_wrong_order.copy()
        self.response_files.sort()

        self.data_files_wrong_order = [
            os.path.join(path, "sensor_data_20190613_170014.pb"),
            os.path.join(path, "sensor_data_20190613_165634.pb"),
        ]
        self.data_files = self.data_files_wrong_order.copy()
        self.data_files.sort()


class TestDataIterator(ProtobufTestsBase):
    """ Check that the DataIterator loads the data/responses and only fails
    to get the sorted values when there is out of order data beyond the window
    size """
    def test_multi_file_response(self):
        """ Test multiple file loading responses doesn't error """
        for resp in DataIterator(self.response_files, responses=True, window=1):
            pass

    def test_multi_file_data_all(self):
        """ Test multiple file loading data doesn't error """
        for data in DataIterator(self.data_files, window=20000):
            pass

    def test_multi_file_data_dm(self):
        """ Test multiple file loading dm data doesn't error """
        for data in DataIterator(self.data_files, data_type=SensorData.MESSAGE_TYPE_DEVICE_MOTION, window=1):
            pass

    def test_multi_file_data_acc(self):
        """ Test multiple file loading acc data doesn't error """
        dataIter = DataIterator(self.data_files, data_type=SensorData.MESSAGE_TYPE_ACCELEROMETER, window=1)
        for data in dataIter:
            pass

    def test_multi_file_data_loc(self):
        """ Test multiple file loading loc data doesn't error """
        dataIter = DataIterator(self.data_files, data_type=SensorData.MESSAGE_TYPE_LOCATION, window=5)
        for data in dataIter:
            pass

    def test_out_of_order_response(self):
        """ Test that responses raises exceptions when out of order and the
        window size isn't big enough to fix it """
        for window in [1, 2, 3, 4, 5]:
            respIter = DataIterator(self.response_files_wrong_order, responses=True, window=window)

            if window < 3:
                with self.assertRaises(NeedLargerWindowError):
                    for resp in respIter:
                        pass
            else:
                for resp in respIter:
                    pass

    def test_out_of_order_data_dm(self):
        """ Test that dm data raises exceptions when out of order """
        with self.assertRaises(NeedLargerWindowError):
            for data in DataIterator(self.data_files_wrong_order, data_type=SensorData.MESSAGE_TYPE_DEVICE_MOTION, window=1000):
                pass

    def test_out_of_order_data_acc(self):
        """ Test that acc data raises exceptions when out of order """
        with self.assertRaises(NeedLargerWindowError):
            for data in DataIterator(self.data_files_wrong_order, data_type=SensorData.MESSAGE_TYPE_ACCELEROMETER, window=1000):
                pass

    def test_out_of_order_data_loc(self):
        """ Test that loc data raises exceptions when out of order """
        with self.assertRaises(NeedLargerWindowError):
            for data in DataIterator(self.data_files_wrong_order, data_type=SensorData.MESSAGE_TYPE_LOCATION, window=5):
                pass

    def test_count_responses(self):
        """ Make sure we have the right amount of response """
        count = 0
        for resp in DataIterator(self.response_files, responses=True, window=1):
            count += 1
        self.assertEqual(count, 3)

    def test_count_data_all(self):
        """ Make sure we have the right amount of data total """
        count = 0
        for resp in DataIterator(self.data_files, window=20000):
            count += 1
        self.assertEqual(count, 25268)

    def test_count_data_loc(self):
        """ Make sure we have the right amount of location data """
        count = 0
        for resp in DataIterator(self.data_files, data_type=SensorData.MESSAGE_TYPE_LOCATION, window=5):
            count += 1
        self.assertEqual(count, 11)

    def test_count_data_dm(self):
        """ Make sure we have the right amount of device motion data """
        count = 0
        for resp in DataIterator(self.data_files, data_type=SensorData.MESSAGE_TYPE_DEVICE_MOTION, window=5):
            count += 1
        self.assertEqual(count, 12621)

    def test_count_data_acc(self):
        """ Make sure we have the right amount of accelerometer data """
        count = 0
        for resp in DataIterator(self.data_files, data_type=SensorData.MESSAGE_TYPE_ACCELEROMETER, window=5):
            count += 1
        self.assertEqual(count, 12634)

    def test_peek_count(self):
        """ Make sure that the peek works and still returns right number """
        count = 0
        respIter = DataIterator(self.response_files, responses=True, window=1)
        lastValue = None
        for resp in respIter:
            self.assertTrue(lastValue is None or resp == lastValue)
            count += 1
            lastValue = respIter.pop()
        self.assertEqual(count, 3)


class TestStateVectorPeekIterator(ProtobufTestsBase):
    def test_state_vector_peek(self):
        """ Check that peeking works and we get the right number of state
        vectors (should be one for each DM value) """
        statePeekIter = StateVectorPeekIterator(self.data_files, window=5)
        lastValue = None
        count = 0

        for state in statePeekIter:
            x = np.array(state["data"], dtype=np.float32)
            x[np.isnan(x)] = 0.0
            statePeekIter.pop()
            self.assertTrue(lastValue is None or x == lastValue)
            count += 1

        self.assertEqual(count, 12621)


if __name__ == "__main__":
    unittest.main()
