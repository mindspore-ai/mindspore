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
# an simple tutorial as follows, more parameters can be setting

if [ $# != 3 ]
then
    echo "Usage: sh run_standalone_train_ascend.sh [CKPT_DIR] [DATA_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export CKPT_DIR=$(get_real_path $1)
export DATA_PATH=$(get_real_path $2)
export DEVICE_ID=$3

python -u ../train.py --ckpt_dir=$CKPT_DIR --data_path=$DATA_PATH \
                      --device_id=$DEVICE_ID --device_target="Ascend" &> train.log &

