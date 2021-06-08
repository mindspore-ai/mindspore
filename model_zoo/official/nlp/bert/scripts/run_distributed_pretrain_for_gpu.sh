#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "======================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR"
echo "for example: bash run_distributed_pretrain.sh 8 40 /path/zh-wiki/ /path/Schema.json"
echo "It is better to use absolute path."
echo "======================================================================================="

RANK_SIZE=4
EPOCH_SIZE=1
DATA_DIR="/data/enwiki/tfrecord"
SCHEMA_DIR="/home/marcel/Mindspore/pretraining_schema.json"

. /home/marcel/Mindspore/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Mindspore/kungfu-mindspore/mindspore)

echo  `date +"%Y-%m-%d %T"`

/home/marcel/.local/bin/mpirun --allow-run-as-root \
    -n $RANK_SIZE \
    --output-filename log_output \
    python run_pretrain.py        \
        --device_target="GPU"      \
        --distribute="true"        \
        --epoch_size=$EPOCH_SIZE    \
        --enable_save_ckpt="true"    \
        --enable_lossscale="true"    \
        --do_shuffle="true"        \
        --enable_data_sink="true"    \
        --data_sink_steps=20        \
        --save_checkpoint_steps=10000  \
        --save_checkpoint_num=1      \
        --data_dir=$DATA_DIR      \
        --train_steps=100 \
        --schema_dir=$SCHEMA_DIR > log.txt 2>&1

echo  `date +"%Y-%m-%d %T"`
