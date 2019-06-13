#!/usr/bin/env python3
import json

if __name__ == "__main__":
    with open("sensor_data_20190607_034536_v2.json") as f:
        print(len(json.loads(f.read())))
