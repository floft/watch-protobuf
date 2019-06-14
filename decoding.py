"""
Shared code for both decoding sensor data and responses
"""


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


def messages_to_json(messages, msg_to_json):
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


def get_enum_str(msg, field_name, enum_int):
    """ Get the human-readable enum value as a string """
    return msg.DESCRIPTOR.fields_by_name[field_name].enum_type.values_by_number[enum_int].name
