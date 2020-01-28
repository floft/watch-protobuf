#!/bin/bash
# Output shapes (from --debug):
# (206, 18) (207, 9) (3, 38)
# (301, 18) (300, 9) (3, 38)
# (300, 18) (301, 9) (0,)
# ...

/usr/bin/time ./process_full.py \
   --dir="/home/garrett/Documents/School/19_Summer_Large/Mobile Activity Study/al-watch-log-upload-api-new" \
   --nums=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
   |& tee output_full.txt
