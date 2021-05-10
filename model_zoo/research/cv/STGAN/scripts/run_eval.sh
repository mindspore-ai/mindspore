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

if [ $# != 4 ]
then
    echo "Usage: sh run_eval.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID] [CHECKPOINT_PATH]"
exit 1
fi

export DATA_PATH=$1
export EXPERIMENT_NAME=$2
export DEVICE_ID=$3
export CHECKPOINT_PATH=$4

python eval.py --dataroot=$DATA_PATH --experiment_name=$EXPERIMENT_NAME \
               --device_id=$DEVICE_ID --ckpt_path=$CHECKPOINT_PATH \
               --platform="Ascend" > eval_log 2>&1 &
