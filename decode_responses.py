#!/usr/bin/env python3
"""
Decode response protobuf into JSON
"""
import os
import sys
import json

from datetime import datetime

from decoding import decode, write_messages
from watch_data_pb2 import PromptResponse


def msg_to_json(msg):
    """ Create JSON from message """
    data = {}

    data["epoch"] = str(datetime.fromtimestamp(msg.epoch))

    if msg.prompt_type == PromptResponse.PROMPT_TYPE_ACTIVITY_QUERY:
        data["label"] = msg.user_activity_label
    else:
        raise NotImplementedError("found unknown message type")

    return json.dumps(data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./decode_responses.py input.pb output.json")
        exit(1)

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    if not os.path.exists(input_fn):
        print("Error: input file does not exist:", input_fn)
        exit(1)
    if os.path.exists(output_fn):
        print("Error: output file exists:", output_fn)
        exit(1)

    write_messages(decode(input_fn, PromptResponse), msg_to_json, output_fn)
