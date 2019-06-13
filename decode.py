#!/usr/bin/env python3
"""
Decode protobuf into JSON
"""
import os
import sys
import json

from datetime import datetime
from google.protobuf import json_format

from watch_data_pb2 import SensorData


def decode(filename):
    result = ""

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

            # Convert to human-readable JSON
            result += json_format.MessageToJson(msg) + ",\n"

            # Note: to get datetime object
            # datetime.fromtimestamp(msg.epoch)

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

    data = decode(input_fn)

    with open(output_fn, "w") as f:
        f.write(data)
