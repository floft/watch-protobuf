#!/usr/bin/env python3
"""
Decode response protobuf into JSON
  - Manually create JSON from desired fields
  - Sort on timestamp
"""
import os
import sys
import json

from datetime import datetime
from google.protobuf import json_format

from watch_data_pb2 import PromptResponse


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
            msg = PromptResponse()
            msg.ParseFromString(data)
            messages.append(msg)

    return messages


def msg_to_json(msg):
    """ Create JSON from message """
    data = {}

    data["epoch"] = str(datetime.fromtimestamp(msg.epoch))

    if msg.prompt_type == PromptResponse.PROMPT_TYPE_ACTIVITY_QUERY:
        data["label"] = msg.user_activity_label
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
        print("Usage: ./decode_response.py input.pb output.json")
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
