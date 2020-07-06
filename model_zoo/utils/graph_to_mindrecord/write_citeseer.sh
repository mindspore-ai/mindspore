#!/bin/bash
SRC_PATH=/tmp/citeseer/dataset
MINDRECORD_PATH=/tmp/citeseer/mindrecord

rm -f $MINDRECORD_PATH/*

python writer.py --mindrecord_script citeseer \
--mindrecord_file "$MINDRECORD_PATH/citeseer_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "$SRC_PATH"
