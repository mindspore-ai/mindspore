#!/bin/bash
rm /tmp/cora/mindrecord/*

python writer.py --mindrecord_script cora \
--mindrecord_file "/tmp/cora/mindrecord/cora_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "/tmp/cora/dataset/cora_content.csv:/tmp/cora/dataset/cora_cites.csv"
