#!/bin/bash
rm responses.{pb,json} sensor_data.{pb,json}
cat responses_*.pb > responses.pb
cat sensor_data_*.pb > sensor_data.pb
../decode_responses.py responses.pb responses.json
../decode_sensor_data.py sensor_data.pb sensor_data.json
