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

from decoding import decode, messages_to_json, get_enum_str
from watch_data_pb2 import SensorData


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
            "calibration_accuracy": get_enum_str(msg, "mag_calibration_acc", mag_calib),
            "x": msg.mag_x,
            "y": msg.mag_y,
            "z": msg.mag_z,
        }
    elif msg.message_type == SensorData.MESSAGE_TYPE_LOCATION:
        data["longitude"] = msg.longitude
        data["latitude"] = msg.latitude
        data["altitude"] = msg.altitude
        data["vertical_accuracy"] = msg.vert_acc
        data["horizontal_accuracy"] = msg.horiz_acc
        data["course"] = msg.course if msg.course != 0.0 else -1.0
        data["speed"] = msg.speed if msg.course != 0.0 else -1.0
        data["floor"] = msg.floor
    elif msg.message_type == SensorData.MESSAGE_TYPE_BATTERY:
        data["bat_level"] = msg.bat_level
        data["bat_state"] = get_enum_str(msg, "bat_state", msg.bat_state)
    else:
        raise NotImplementedError("found unknown message type")

    return json.dumps(data)


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

    data = messages_to_json(decode(input_fn, SensorData), msg_to_json)

    with open(output_fn, "w") as f:
        f.write(data)
