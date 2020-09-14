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

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "sh run_standalone_train.sh DEVICE_ID EPOCH_SIZE DATA_PATH"
echo "for example: sh run_standalone_train.sh 0 52 /path/ende-l128-mindrecord00"
echo "It is better to use absolute path."
echo "=============================================================================================================="

rm -rf run_standalone_train
mkdir run_standalone_train
cp -rf ./src/ train.py ./run_standalone_train
cd run_standalone_train || exit

export DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_PATH=$3

python train.py  \
    --distribute="false" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --enable_data_sink="false" \
    --checkpoint_path="" \
    --save_checkpoint_steps=2500 \
    --save_checkpoint_num=30 \
    --data_path=$DATA_PATH > log.txt 2>&1 &
cd ..