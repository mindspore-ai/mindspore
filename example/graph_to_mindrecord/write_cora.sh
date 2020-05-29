#!/bin/bash
SRC_PATH=/tmp/cora/dataset
MINDRECORD_PATH=/tmp/cora/mindrecord

rm -f $MINDRECORD_PATH/*

python writer.py --mindrecord_script cora \
--mindrecord_file "$MINDRECORD_PATH/cora_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "$SRC_PATH/cora_content.csv:$SRC_PATH/cora_cites.csv"
