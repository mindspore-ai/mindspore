#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
if [ $# != 2 ]
then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]."
    exit 1
fi

export RANK_TABLE_FILE=$1
export DEVICE_NUM=8
export RANK_SIZE=8

for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    echo "start distributed training for rank $RANK_ID, device $DEVICE_ID"
    python train.py --device_id=$DEVICE_ID --train_data_dir=$2 --run_distribute=True &>train$i.log & 
done