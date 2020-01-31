#!/bin/bash
# Output shapes (from --debug):
# x shapes: (301, 18) (300, 9) (2, 38) (301,) (300,) (2,)
# x shapes: (300, 18) (301, 9) (0,) (300,) (301,) (0,)
# x shapes: (301, 18) (300, 9) (0,) (301,) (300,) (0,)
# ...
# ./process_full.py \
#    --dir="/home/garrett/Documents/School/19_Summer_Large/Mobile Activity Study/al-watch-log-upload-api-new" \
#    --nums=1 --jobs=1 --debug

/usr/bin/time ./process_full.py \
   --dir="/home/garrett/Documents/School/19_Summer_Large/Mobile Activity Study/al-watch-log-upload-api-new" \
   --nums=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
   |& tee output_full.txt
