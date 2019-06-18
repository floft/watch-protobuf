#!/usr/bin/env python3
"""
Convert the sensor data lat/lon/alt to a KML file to visualize in Google Earth

pip install --user fastkml
sudo pacman -S python-shapely
"""
import os
import sys

from datetime import datetime
from fastkml import kml, styles, geometry

from decoding import decode
from watch_data_pb2 import SensorData


def write_kml(messages, output_filename):
    """ Sort messages on timestamp, convert to KML and write to disk """
    # Sort since when saving to a file on the watch, they may be out of order
    messages.sort(key=lambda x: x.epoch)

    # Create KML file
    k = kml.KML()
    ns = '{http://www.opengis.net/kml/2.2}'
    d = kml.Document(ns, 'watch-loc-data', 'Watch location data')
    k.append(d)
    f = kml.Folder(ns, 'all-data', 'All location data')
    d.append(f)
    s = [kml.Style(ns, 'styles', [
        styles.LineStyle(ns, 'linestyle', 'FF0000FF', width=2),
        styles.PolyStyle(ns, 'polystyle', '00FFFFFF'),
    ])]

    i = 0
    pt_prev = None
    ts_prev = None

    for msg in messages:

        if msg.message_type == SensorData.MESSAGE_TYPE_LOCATION:
            # Skip if invalid lat/lon/alt value
            if msg.longitude == 0.0 and msg.latitude == 0.0 and msg.horiz_acc == 0.0:
                continue
            if msg.altitude == 0.0 and msg.vert_acc == 0.0:
                continue

            ts = datetime.fromtimestamp(msg.epoch)
            pt = (msg.longitude, msg.latitude, msg.altitude)

            # We're drawing lines between points, so skip the first point
            if i != 0:
                p = kml.Placemark(ns, 'point-'+str(i), 'point-'+str(i), styles=s)
                p.geometry = geometry.Geometry(ns, 'geometry-'+str(i),
                    geometry.Polygon([pt_prev, pt, pt, pt_prev]),
                    altitude_mode='absolute')
                p.begin = ts_prev
                p.end = ts
                f.append(p)

            i += 1
            pt_prev = pt
            ts_prev = ts

    with open(output_filename, "w") as f:
        f.write(k.to_string(prettyprint=True))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./kml.py input.pb output.kml")
        exit(1)

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    if not os.path.exists(input_fn):
        print("Error: input file does not exist:", input_fn)
        exit(1)
    if os.path.exists(output_fn):
        print("Error: output file exists:", output_fn)
        exit(1)

    write_kml(decode(input_fn, SensorData), output_fn)
