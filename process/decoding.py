"""
Decode the protobuf files
"""
from datetime import datetime

from watch_data_pb2 import SensorData, PromptResponse


def decode(filename, message_type):
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
            msg = message_type()
            msg.ParseFromString(data)
            messages.append(msg)

    return messages


def get_enum_str(msg, field_name, enum_int):
    """ Get the human-readable enum value as a string """
    return msg.DESCRIPTOR.fields_by_name[field_name].enum_type.values_by_number[enum_int].name


def parse_data(msg, ignore_all_except=None):
    """ Parse a data message """
    # Skip all except the specified one, if it is specified
    if ignore_all_except is not None and msg.message_type != ignore_all_except:
        return None

    data = {}
    data["epoch"] = msg.epoch
    #data["message_type"] = get_enum_str(msg, "message_type", msg.message_type)

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
        data["heading"] = msg.heading if msg.heading != 0.0 else None

        if msg.mag_calibration_acc == SensorData.MAG_CALIBRATION_UNSPECIFIED:
            data["magnetic_field"] = {
                "calibration_accuracy":
                    get_enum_str(msg, "mag_calibration_acc", SensorData.MAG_CALIBRATION_UNCALIBRATED),
                "x": None,
                "y": None,
                "z": None,
            }
        else:
            data["magnetic_field"] = {
                "calibration_accuracy": get_enum_str(msg, "mag_calibration_acc", msg.mag_calibration_acc),
                "x": msg.mag_x,
                "y": msg.mag_y,
                "z": msg.mag_z,
            }
    elif msg.message_type == SensorData.MESSAGE_TYPE_LOCATION:
        # Note: default values for numeric types are 0 with proto3, so we can't
        # differentiate 0.0 from unspecified. However, it's highly unlikely that
        # multiple values are exactly 0.0, so we'll use that to determine if
        # the values should be valid.
        if msg.longitude == 0.0 and msg.latitude == 0.0 and msg.horiz_acc == 0.0:
            data["longitude"] = None
            data["latitude"] = None
            data["horizontal_accuracy"] = None
        else:
            data["longitude"] = msg.longitude
            data["latitude"] = msg.latitude
            data["horizontal_accuracy"] = msg.horiz_acc

        if msg.altitude == 0.0 and msg.vert_acc == 0.0:
            data["altitude"] = None
            data["vertical_accuracy"] = None
        else:
            data["altitude"] = msg.altitude
            data["vertical_accuracy"] = msg.vert_acc

        data["course"] = msg.course if msg.course != 0.0 else None
        data["speed"] = msg.speed if msg.speed != 0.0 else None
        data["floor"] = msg.floor if msg.floor != 0 else None
    elif msg.message_type == SensorData.MESSAGE_TYPE_BATTERY:
        data["bat_level"] = msg.bat_level
        data["bat_state"] = get_enum_str(msg, "bat_state", msg.bat_state)
    else:
        raise NotImplementedError("found unknown message type")

    return data


def parse_response(msg):
    """ Parse a response message """
    data = {}
    data["epoch"] = msg.epoch

    if msg.prompt_type == PromptResponse.PROMPT_TYPE_ACTIVITY_QUERY:
        data["label"] = msg.user_activity_label
    else:
        raise NotImplementedError("found unknown message type")

    return data
