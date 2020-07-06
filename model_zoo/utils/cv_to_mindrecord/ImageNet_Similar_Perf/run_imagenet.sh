#!/bin/bash
rm /tmp/imagenet/mr/*

python writer.py --mindrecord_script imagenet \
--mindrecord_file "/tmp/imagenet/mr/m" \
--mindrecord_partitions 16 \
--label_file "/tmp/imagenet/label.txt" \
--image_dir "/tmp/imagenet/jpeg"
