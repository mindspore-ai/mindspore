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

if [ $# != 0 ]
then
    echo "Usage: sh run_train.sh"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi


dataset_type='imagenet'


ulimit -u unlimited
export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

echo "start training for device $DEVICE_ID"
python train.py --device_id=$DEVICE_ID --dataset_name=$dataset_type> log 2>&1 &