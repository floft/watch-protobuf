"""
Reverse geocoding and caching code
"""
import json
import socket
import urllib.request

from lru import LRU
from urllib.error import HTTPError, URLError


cache = LRU(1000)


def reverse_geocode(lat, lon, timeout=300):
    """
    Get location information about a GPS coordinate (lat, lon) with
    a local instance of Open Street Maps Nominatim

    This assumes you have a Nominatim server running on 7070 as described
    in README.md

    We use a large timeout since sometimes postgresql decides to autovacuum the
    database which takes a bit of time. Though, really, we should probably just
    wait till it's done doing that.

    We could use geopy, but it seems to download this in the "json" format
    hard-coded, but we need category, etc. information only available in the
    other formats such as jsonv2.

    See hard-coded format in geopy:
    https://github.com/geopy/geopy/blob/85ccae74f2e8011c89d2e78d941b5df414ab99d1/geopy/geocoders/osm.py#L453

    See example reverse lookup JSON with "jsonv2" format:
    https://nominatim.org/release-docs/develop/api/Reverse/#example-with-formatjsonv2

    Code for handling timeouts:
    https://stackoverflow.com/q/8763451
    """
    # See if we can get value from cache first
    key = (lat, lon)

    if key in cache:
        return cache[key]

    # If not, actually load from Nominatim
    url = "http://localhost:7070/reverse?format=jsonv2&lat=" \
        + str(lat) + "&lon=" + str(lon)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as con:
            result = json.loads(con.read().decode("utf-8"))

            # Cache for next time
            cache[key] = result

            return result
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
