watch-protobuf
==============

 - Download files from watch
 - Compile protobuf definition: `protoc watch-data.proto --python_out=.`
 - Decode responses: `cat responses_*.pb > responses.pb; python3 decode_responses.py responses.pb responses.json`
 - Decode sensor data: `cat sensor_data_*.pb > sensor_data.pb; python3 decode_sensor_data.py sensor_data.pb sensor_data.json`

New processing code is in *process/* (self contained). It depends on:
tqdm, absl, tensorflow,
