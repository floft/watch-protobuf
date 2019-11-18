process watch data to .tfrecord files
=====================================
Dependencies:

    pip3 install --user tensorflow sklearn tqdm absl lru-dict

Setting up Nominatim server (download
[region of interest](https://download.geofabrik.de/) and see Docker container
[documentation](https://github.com/mediagis/nominatim-docker)):

    git clone https://github.com/mediagis/nominatim-docker
    cd nominatim-docker

    # Note: if you only have 16 GiB of RAM, then try adding
    # --osm2pgsql-cache 1000 to <version>/init.sh

    mkdir data
    sudo chattr +C data  # if using btrfs, disable copy-on-write (COW)
    cd data
    wget https://download.geofabrik.de/north-america/us-west-191101.osm.pbf
    wget https://download.geofabrik.de/north-america/us-west-191101.osm.pbf.md5
    md5sum -c us-west-191101.osm.pbf.md5
    cd ..

    cd 3.4
    sudo systemctl start docker
    sudo docker build -t nominatim .
    sudo docker run -t -v /path/to/nominatim-docker/data:/data nominatim  sh /app/init.sh /data/us-west-191101.osm.pbf postgresdata 8
    sudo docker run --restart=always -p 6432:5432 -p 7070:8080 -d --name nominatim -v /home/garrett/Documents/Github/nominatim-docker/data/postgresdata:/var/lib/postgresql/11/main nominatim bash /app/start.sh

Then, to process a few watch files:

    ./process.py --dir=/path/to/watch/files --nums=1,2,3
