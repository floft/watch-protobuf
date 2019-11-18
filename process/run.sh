#!/bin/bash
/usr/bin/time ./process.py \
    --dir="/home/garrett/Documents/School/19_Summer_Large/Mobile Activity Study/al-watch-log-upload-api-new" \
    --nums=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    | tee output.txt
