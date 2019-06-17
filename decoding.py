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


def write_messages(messages, msg_to_json_fn, output_filename):
    """ Sort messages on timestamp, convert to JSON and write to disk """
    # Sort since when saving to a file on the watch, they may be out of order
    messages.sort(key=lambda x: x.epoch)

    # Output JSON
    with open(output_filename, "w") as f:
        f.write("[")

        for i, msg in enumerate(messages):
            f.write(msg_to_json_fn(msg))

            # Invalid JSON if we have an extra comma at the end
            if i != len(messages)-1:
                f.write(",\n")

        f.write("]\n")


def get_enum_str(msg, field_name, enum_int):
    """ Get the human-readable enum value as a string """
    return msg.DESCRIPTOR.fields_by_name[field_name].enum_type.values_by_number[enum_int].name
