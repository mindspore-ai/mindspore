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

PRETRAINED_CKPT=""

ulimit -m unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash scripts/run_standalone_train.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]"
    echo "============================================================"
    echo "[DATA_PATH]: The path to the train and evaluation datasets."
    echo "[SAVE_DIR]: The path to save files generated during training."
    echo "[PRETRAINDE_CKPT]: (optional) The path to the checkpoint file."
    echo "============================================================"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# -ge 2 ]
then
    DATA_PATH=$(get_real_path $1)
    SAVE_DIR=$(get_real_path $2)

    if [ ! -d $DATA_PATH ]
    then
        echo "error: DATA_PATH=$DATA_PATH is not a directory"
    exit 1
    fi
fi

if [ $# -ge 3 ]
then
    PRETRAINED_CKPT=$(get_real_path $3)
    if [ ! -f $PRETRAINED_CKPT ]
    then
        echo "error: PRETRAINED_CKPT=$PRETRAINED_CKPT is not a file"
    exit 1
    fi
fi

python train.py \
  --data_path=$DATA_PATH \
  --pretrained_ckpt=$PRETRAINED_CKPT \
  --save_dir=$SAVE_DIR > train.log 2>&1 &

echo 'running'
