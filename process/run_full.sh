#!/bin/bash
# Output shapes (from --debug):
# (206, 18) (207, 9) (3, 38)
# (301, 18) (300, 9) (3, 38)
# (300, 18) (301, 9) (0,)
# ...

/usr/bin/time ./process_full.py \
   --dir="/home/garrett/Documents/School/19_Summer_Large/Mobile Activity Study/al-watch-log-upload-api-new" \
   --nums=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --split --downsample=10 \
   |& tee output_full.txt

# Split into train-valid-test, normalize, clean -- limits jobs since otherwise
# we run out of memory (despite having 32 GiB)
# /usr/bin/time ./process_full2.py \
#    --nums=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --jobs=1 \
#    |& tee output_full2.txt
