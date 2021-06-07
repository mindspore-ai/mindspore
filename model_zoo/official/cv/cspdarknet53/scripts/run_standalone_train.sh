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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR] (option)[PATH_CHECKPOINT]"
exit 1
fi

export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$1

DATA_DIR=$2
PATH_CHECKPOINT=""
if [ $# == 3 ]
then
  PATH_CHECKPOINT=$3
fi

python train.py \
    --is_distributed=0 \
    --pretrained=$PATH_CHECKPOINT \
    --data_dir=$DATA_DIR > log.txt 2>&1 &

