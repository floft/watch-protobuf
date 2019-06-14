#!/usr/bin/env python3
"""
Decode protobuf into JSON
  - Manually create JSON from desired fields
  - Sort on timestamp
"""
import os
import sys
import json

from datetime import datetime
from google.protobuf import json_format

from watch_data_pb2 import SensorData


def decode(filename):
    """ Decode protobuf messages from file """
    messages = []

    with open(filename, "rb") as f:
        while True:
            # Get size of message, then read that many bytes
            size = f.read(2)

            if size == b"":  # eof
                break

            size = int.from_bytes(size, "little")
            data = f.read(size)

            # Create message from read bytes
            msg = SensorData()
            msg.ParseFromString(data)
            messages.append(msg)

    return messages


def msg_to_json(msg):
    """ Create JSON from message """
    data = {}

    data["epoch"] = str(datetime.fromtimestamp(msg.epoch))

    if msg.message_type == SensorData.MESSAGE_TYPE_ACCELEROMETER:
        data["raw_acceleration"] = {
            "x": msg.raw_accel_x,
            "y": msg.raw_accel_y,
            "z": msg.raw_accel_z,
        }
    elif msg.message_type == SensorData.MESSAGE_TYPE_DEVICE_MOTION:
        data["attitude"] = {
            "roll": msg.roll,
            "pitch": msg.pitch,
            "yaw": msg.yaw,
        }
        data["rotation_rate"] = {
            "x": msg.rot_rate_x,
            "y": msg.rot_rate_y,
            "z": msg.rot_rate_z,
        }
        data["user_acceleration"] = {
            "x": msg.user_accel_x,
            "y": msg.user_accel_y,
            "z": msg.user_accel_z,
        }
        data["gravity"] = {
            "x": msg.grav_x,
            "y": msg.grav_y,
            "z": msg.grav_z,
        }
        data["heading"] = msg.heading if msg.course != 0.0 else -1.0

        if msg.mag_calibration_acc == SensorData.MAG_CALIBRATION_UNSPECIFIED:
            mag_calib = SensorData.MAG_CALIBRATION_UNCALIBRATED
        else:
            mag_calib = msg.mag_calibration_acc

        data["magnetic_field"] = {
            "calibration_accuracy": msg.DESCRIPTOR.fields_by_name["mag_calibration_acc"].enum_type.values_by_number[mag_calib].name,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    elif msg.message_type == SensorData.MESSAGE_TYPE_LOCATION:
        data["coordinate"] = {
            "longitude": msg.longitude,
            "latitude": msg.latitude,
        }
        data["altitude"] = msg.altitude
        data["vertical_accuracy"] = msg.vert_acc
        data["horizontal_accuracy"] = msg.horiz_acc
        data["course"] = msg.course if msg.course != 0.0 else -1.0
        data["speed"] = msg.speed if msg.course != 0.0 else -1.0
        data["floor"] = msg.floor
    elif msg.message_type == SensorData.MESSAGE_TYPE_BATTERY:
        data["level"] = msg.bat_level
        data["state"] = msg.DESCRIPTOR.fields_by_name["bat_state"].enum_type.values_by_number[msg.bat_state].name
    else:
        raise NotImplementedError("found unknown message type")

    return json.dumps(data)


def messages_to_json(messages):
    """ Convert list of messages to JSON and sort on timestamp """
    # Sort since when saving to a file on the watch, they may be out of order
    messages.sort(key=lambda x: x.epoch)

    # Output JSON
    result = ""

    for msg in messages:
        result += msg_to_json(msg) + ",\n"

    # Remove the last comma and new line since last comma is invalid JSON
    if result != "":
        result = "[" + result[:-2] + "]"

    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./decode.py input.pb output.json")
        exit(1)

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    if not os.path.exists(input_fn):
        print("Error: input file does not exist:", input_fn)
        exit(1)
    if os.path.exists(output_fn):
        print("Error: output file exists:", output_fn)
        exit(1)

    data = messages_to_json(decode(input_fn))

    with open(output_fn, "w") as f:
        f.write(data)
