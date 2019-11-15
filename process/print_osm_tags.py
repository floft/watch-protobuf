#!/usr/bin/env python3
"""
Get categories and types for reverse geocoding lookup

See: https://taginfo.openstreetmap.org/keys

Run: ./print_osm_tags.py | tee osm_tags.py
"""
import time
import json
import socket
import urllib.request


from urllib.error import HTTPError, URLError


def get_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as con:
            return json.loads(con.read().decode("utf-8"))
    except HTTPError as error:
        print("Warning:", error, "loading", url)
    except URLError as error:
        if isinstance(error.reason, socket.timeout):
            print("Warning: socket timed out loading", url)
        else:
            print("Warning: unknown error loading", url)
    except socket.timeout:
        print("Warning: socket timed out loading", url)

    return None


def get_keys(key, max_results=100, min_fraction=0.0001):
    url = "https://taginfo.openstreetmap.org/api/4/key/values?" \
        + "key=" + str(key) + "&" \
        + "filter=all&lang=en&sortname=count&sortorder=desc&page=1&rp=" \
        + str(max_results) + "&qtype=value&format=json_pretty"
    results = get_json(url)

    keys = []

    if results is not None and "data" in results:
        for result in results["data"]:
            if result["fraction"] >= min_fraction:
                keys.append(result["value"])

    return keys


def print_str_list(array, array_name):
    print(array_name, "= [")
    for v in array:
        print("    \""+v+"\",")
    print("]")


if __name__ == "__main__":
    categories = [
        "building",
        "highway",
        "natural",
        "surface",
        "landuse",
        "power",
        "waterway",
        "amenity",
        "oneway",
        "wall",
        "service",
        "place",
        "shop",
        "barrier",
        "crossing",
        "tourism",
        "footway",
        "water",
        "bridge",
    ]
    keys = []

    for category in categories:
        keys += get_keys(category)

        # Don't overwhelm their servers
        time.sleep(1)

    # Tags/keys aren't necessarily unique, so sort/unique them
    categories = sorted(set(categories))
    keys = sorted(set(keys))

    # Output
    print_str_list(categories, "categories")
    print_str_list(keys, "keys")
