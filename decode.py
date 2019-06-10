#!/usr/bin/env python3
"""
Decode protobuf into JSON
"""
import os
import sys
import json

from datetime import datetime
from google.protobuf import json_format

import watch_pb2


def decode(filename):
    result = ""

    with open(filename, "rb") as f:
        while True:
            # Get size of message, then read that many bytes
            size = f.read(2)

            if size == b"":  # eof
                break

            size = int.from_bytes(size, "big")
            data = f.read(size)

            # Create message from read bytes
            msg = watch_pb2.SensorData()
            msg.ParseFromString(data)

            # Determine what type of message
            result += json_format.MessageToJson(msg) + "\n"

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
