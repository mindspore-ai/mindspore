#!/bin/bash
MINDRECORD_PATH=/tmp/sns

rm -f $MINDRECORD_PATH/*

python writer.py --mindrecord_script sns \
--mindrecord_file "$MINDRECORD_PATH/sns" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 14 \
--mindrecord_page_size_by_bit 15
