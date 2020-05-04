#!/bin/bash
rm /tmp/citeseer/mindrecord/*

python writer.py --mindrecord_script citeseer \
--mindrecord_file "/tmp/citeseer/mindrecord/citeseer_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "/tmp/citeseer/dataset/citeseer.content:/tmp/citeseer/dataset/citeseer.cites"
