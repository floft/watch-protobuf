#!/usr/bin/env python3
"""
Load the JSON data and put into one of two protobuf formats
Compare sizes
"""
import json

from datetime import datetime

# WARNING: inconvenient, but apparently uncommenting both gives an error, so
# run them one at a time...
import watch_nonest_pb2
#import watch_nest_pb2
#import watch_nest2_pb2


def lsplit(string_to_split, split_str, max_number=None):
    """ Split string from left but with max number of splits
    Example: lsplit("1,2,3", ",", 2) will return ["1", "2,3"] """
    parts = string_to_split.split(split_str)

    if max_number is not None and len(parts) > max_number:
        assert max_number > 0
        last = split_str.join(parts[max_number-1:])
        return parts[:max_number-1] + [last]
    else:
        return parts


def encode_nonest(data):
    output = b""

    for ts, id, obj in data:
        msg = watch_nonest_pb2.SensorData()
        msg.timestamp = str(ts)

        # Note: these could have been stored as floats in JSON (without quotes)
        if id == "ac":
            msg.seconds_since_boot = float(obj["seconds_since_boot"])
            msg.raw_accel_x = float(obj["acceleration"]["x"])
            msg.raw_accel_y = float(obj["acceleration"]["y"])
            msg.raw_accel_z = float(obj["acceleration"]["z"])
        elif id == "dm":
            msg.seconds_since_boot = float(obj["seconds_since_boot"])
            msg.roll = float(obj["attitude"]["roll"])
            msg.pitch = float(obj["attitude"]["pitch"])
            msg.yaw = float(obj["attitude"]["yaw"])
            msg.rot_rate_x = float(obj["rotation_rate"]["x"])
            msg.rot_rate_y = float(obj["rotation_rate"]["y"])
            msg.rot_rate_z = float(obj["rotation_rate"]["z"])
            if float(obj["heading"]) != -1:
                msg.heading = float(obj["heading"])
            msg.user_accel_x = float(obj["user_acceleration"]["x"])
            msg.user_accel_y = float(obj["user_acceleration"]["y"])
            msg.user_accel_z = float(obj["user_acceleration"]["z"])
            msg.grav_x = float(obj["gravity"]["x"])
            msg.grav_y = float(obj["gravity"]["y"])
            msg.grav_z = float(obj["gravity"]["z"])

            calib = obj["magnetic_field"]["calibration_accuracy"]
            if calib != "uncalibrated":
                if calib == "uncalibrated":
                    msg.mag_calibration_acc = watch_nonest_pb2.SensorData.UNCALIBRATED
                elif calib == "low":
                    msg.mag_calibration_acc = watch_nonest_pb2.SensorData.LOW
                elif calib == "medium":
                    msg.mag_calibration_acc = watch_nonest_pb2.SensorData.MEDIUM
                elif calib == "high":
                    msg.mag_calibration_acc = watch_nonest_pb2.SensorData.HIGH
                else:
                    raise NotImplementedError("found unknown calibration_accuracy")
                msg.mag_x = float(obj["magnetic_field"]["x"])
                msg.mag_y = float(obj["magnetic_field"]["y"])
                msg.mag_z = float(obj["magnetic_field"]["z"])
        elif id == "lc":
            if float(obj["vertical_accuracy"]) >= 0:
                msg.altitude = float(obj["altitude"])
            if float(obj["horizontal_accuracy"]) >= 0:
                msg.longitude = float(obj["coordinate"]["longitude"])
                msg.latitude = float(obj["coordinate"]["latitude"])
            if float(obj["course"]) != -1:
                msg.course = float(obj["course"])
            if float(obj["speed"]) != -1:
                msg.speed = float(obj["speed"])
            if float(obj["horizontal_accuracy"]) >= 0:
                msg.horiz_acc = float(obj["horizontal_accuracy"])
            if float(obj["vertical_accuracy"]) >= 0:
                msg.vert_acc = float(obj["vertical_accuracy"])
            if obj["floor"] is not None:
                msg.floor = float(obj["floor"])
        elif id == "ba":
            msg.bat_level = float(obj["level"])

            if obj["state"] == "charging":
                msg.bat_state = watch_nonest_pb2.SensorData.CHARGING
            else:
                raise NotImplementedError("other battery states not implemented yet")
        else:
            raise NotImplementedError("found unknown message type "+id)

        msg_str = msg.SerializeToString()
        msg_len = len(msg_str)

        # Network byte order is big endian, though probably both platforms we
        # care about are actually little endian...
        output += msg_len.to_bytes(2, "big") + msg_str

    return output


def encode_nest(data):
    output = b""

    for ts, id, obj in data:
        msg = watch_nest_pb2.SensorData()
        msg.timestamp = str(ts)

        # Note: these could have been stored as floats in JSON (without quotes)
        if id == "ac":
            msg.seconds_since_boot = float(obj["seconds_since_boot"])
            msg.raw_accel.x = float(obj["acceleration"]["x"])
            msg.raw_accel.y = float(obj["acceleration"]["y"])
            msg.raw_accel.z = float(obj["acceleration"]["z"])
        elif id == "dm":
            msg.seconds_since_boot = float(obj["seconds_since_boot"])
            msg.motion.roll = float(obj["attitude"]["roll"])
            msg.motion.pitch = float(obj["attitude"]["pitch"])
            msg.motion.yaw = float(obj["attitude"]["yaw"])
            msg.motion.rot_rate.x = float(obj["rotation_rate"]["x"])
            msg.motion.rot_rate.y = float(obj["rotation_rate"]["y"])
            msg.motion.rot_rate.z = float(obj["rotation_rate"]["z"])
            if float(obj["heading"]) != -1:
                msg.motion.heading = float(obj["heading"])
            msg.motion.user_accel.x = float(obj["user_acceleration"]["x"])
            msg.motion.user_accel.y = float(obj["user_acceleration"]["y"])
            msg.motion.user_accel.z = float(obj["user_acceleration"]["z"])
            msg.motion.gravity.x = float(obj["gravity"]["x"])
            msg.motion.gravity.y = float(obj["gravity"]["y"])
            msg.motion.gravity.z = float(obj["gravity"]["z"])

            calib = obj["magnetic_field"]["calibration_accuracy"]
            if calib != "uncalibrated":
                if calib == "uncalibrated":
                    msg.motion.mag_calibration_acc = watch_nest_pb2.Motion.UNCALIBRATED
                elif calib == "low":
                    msg.motion.mag_calibration_acc = watch_nest_pb2.Motion.LOW
                elif calib == "medium":
                    msg.motion.mag_calibration_acc = watch_nest_pb2.Motion.MEDIUM
                elif calib == "high":
                    msg.motion.mag_calibration_acc = watch_nest_pb2.Motion.HIGH
                else:
                    raise NotImplementedError("found unknown calibration_accuracy")
                msg.motion.mag.x = float(obj["magnetic_field"]["x"])
                msg.motion.mag.y = float(obj["magnetic_field"]["y"])
                msg.motion.mag.z = float(obj["magnetic_field"]["z"])
        elif id == "lc":
            if float(obj["vertical_accuracy"]) >= 0:
                msg.gps.altitude = float(obj["altitude"])
            if float(obj["horizontal_accuracy"]) >= 0:
                msg.gps.longitude = float(obj["coordinate"]["longitude"])
                msg.gps.latitude = float(obj["coordinate"]["latitude"])
            if float(obj["course"]) != -1:
                msg.gps.course = float(obj["course"])
            if float(obj["speed"]) != -1:
                msg.gps.speed = float(obj["speed"])
            if float(obj["horizontal_accuracy"]) >= 0:
                msg.gps.horiz_acc = float(obj["horizontal_accuracy"])
            if float(obj["vertical_accuracy"]) >= 0:
                msg.gps.vert_acc = float(obj["vertical_accuracy"])
            if obj["floor"] is not None:
                msg.gps.floor = float(obj["floor"])
        elif id == "ba":
            msg.bat.level = float(obj["level"])

            if obj["state"] == "charging":
                msg.bat.state = watch_nest_pb2.Battery.CHARGING
            else:
                raise NotImplementedError("other battery states not implemented yet")
        else:
            raise NotImplementedError("found unknown message type "+id)

        msg_str = msg.SerializeToString()
        msg_len = len(msg_str)

        # Network byte order is big endian, though probably both platforms we
        # care about are actually little endian...
        output += msg_len.to_bytes(2, "big") + msg_str

    return output


def encode_nest2(data):
    output = b""

    for ts, id, obj in data:
        msg = watch_nest2_pb2.SensorData()
        msg.timestamp = str(ts)

        # Note: these could have been stored as floats in JSON (without quotes)
        if id == "ac":
            msg.seconds_since_boot = float(obj["seconds_since_boot"])
            msg.raw_accel_x = float(obj["acceleration"]["x"])
            msg.raw_accel_y = float(obj["acceleration"]["y"])
            msg.raw_accel_z = float(obj["acceleration"]["z"])
        elif id == "dm":
            msg.seconds_since_boot = float(obj["seconds_since_boot"])
            msg.motion.roll = float(obj["attitude"]["roll"])
            msg.motion.pitch = float(obj["attitude"]["pitch"])
            msg.motion.yaw = float(obj["attitude"]["yaw"])
            msg.motion.rot_rate_x = float(obj["rotation_rate"]["x"])
            msg.motion.rot_rate_y = float(obj["rotation_rate"]["y"])
            msg.motion.rot_rate_z = float(obj["rotation_rate"]["z"])
            msg.motion.user_accel_x = float(obj["user_acceleration"]["x"])
            msg.motion.user_accel_y = float(obj["user_acceleration"]["y"])
            msg.motion.user_accel_z = float(obj["user_acceleration"]["z"])
            msg.motion.grav_x = float(obj["gravity"]["x"])
            msg.motion.grav_y = float(obj["gravity"]["y"])
            msg.motion.grav_z = float(obj["gravity"]["z"])
            if float(obj["heading"]) != -1:
                msg.motion.heading = float(obj["heading"])

            calib = obj["magnetic_field"]["calibration_accuracy"]
            if calib != "uncalibrated":
                if calib == "uncalibrated":
                    msg.motion.mag_calibration_acc = watch_nest2_pb2.Motion.UNCALIBRATED
                elif calib == "low":
                    msg.motion.mag_calibration_acc = watch_nest2_pb2.Motion.LOW
                elif calib == "medium":
                    msg.motion.mag_calibration_acc = watch_nest2_pb2.Motion.MEDIUM
                elif calib == "high":
                    msg.motion.mag_calibration_acc = watch_nest2_pb2.Motion.HIGH
                else:
                    raise NotImplementedError("found unknown calibration_accuracy")
                msg.motion.mag_x = float(obj["magnetic_field"]["x"])
                msg.motion.mag_y = float(obj["magnetic_field"]["y"])
                msg.motion.mag_z = float(obj["magnetic_field"]["z"])
        elif id == "lc":
            if float(obj["vertical_accuracy"]) >= 0:
                msg.gps.altitude = float(obj["altitude"])
            if float(obj["horizontal_accuracy"]) >= 0:
                msg.gps.longitude = float(obj["coordinate"]["longitude"])
                msg.gps.latitude = float(obj["coordinate"]["latitude"])
            if float(obj["horizontal_accuracy"]) >= 0:
                msg.gps.horiz_acc = float(obj["horizontal_accuracy"])
            if float(obj["vertical_accuracy"]) >= 0:
                msg.gps.vert_acc = float(obj["vertical_accuracy"])
            if float(obj["course"]) != -1:
                msg.gps.course = float(obj["course"])
            if float(obj["speed"]) != -1:
                msg.gps.speed = float(obj["speed"])
            if obj["floor"] is not None:
                msg.gps.floor = float(obj["floor"])
        elif id == "ba":
            msg.bat.level = float(obj["level"])

            if obj["state"] == "charging":
                msg.bat.state = watch_nest2_pb2.Battery.CHARGING
            else:
                raise NotImplementedError("other battery states not implemented yet")
        else:
            raise NotImplementedError("found unknown message type "+id)

        msg_str = msg.SerializeToString()
        msg_len = len(msg_str)

        # Network byte order is big endian, though probably both platforms we
        # care about are actually little endian...
        output += msg_len.to_bytes(2, "big") + msg_str

    return output


def load_json(filename):
    data = []

    with open(filename) as f:
        for line in f:
            timestamp, type_id, json_data = lsplit(line.strip(), ",", 3)
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            json_data = json.loads(json_data)
            data.append((timestamp, type_id, json_data))

    return data


if __name__ == "__main__":
    data = load_json("sensor_data_20190607_034536.log")
    print("No nest:", len(encode_nonest(data)), "bytes")
    #print("Nest:", len(encode_nest(data)), "bytes")
    #print("Nest 2:", len(encode_nest2(data)), "bytes")
