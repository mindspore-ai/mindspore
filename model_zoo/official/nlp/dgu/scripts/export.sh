#!/bin/bash

python export.py --device_id=0 \
        --batch_size=32  \
        --number_labels=26  \
        --ckpt_file=/home/ma-user/work/ckpt/atis_intent/0.9791666666666666_atis_intent-11_155.ckpt  \
        --file_name=atis_intent.mindir  \
        --file_format=MINDIR  \
        --device_target=Ascend
