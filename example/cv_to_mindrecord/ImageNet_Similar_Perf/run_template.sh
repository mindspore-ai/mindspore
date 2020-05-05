#!/bin/bash
rm /tmp/template/*

python writer.py --mindrecord_script template \
--mindrecord_file "/tmp/template/m" \
--mindrecord_partitions 4
