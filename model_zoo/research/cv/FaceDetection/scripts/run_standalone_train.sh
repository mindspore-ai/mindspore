#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash run_standalone_train.sh [PLATFORM] [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]"
    echo "   or: bash run_standalone_train.sh [PLATFORM] [MINDRECORD_FILE] [USE_DEVICE_ID]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

export PYTHONPATH=${dirname_path}:$PYTHONPATH

export RANK_SIZE=1

SCRIPT_NAME='train.py'

ulimit -c unlimited

PLATFORM=$1
MINDRECORD_FILE=$(get_real_path $2)
USE_DEVICE_ID=$3
PRETRAINED_BACKBONE=''

if [ $# == 4 ]
then
    PRETRAINED_BACKBONE=$(get_real_path $4)
    if [ ! -f $PRETRAINED_BACKBONE ]
    then
        echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
    exit 1
    fi
fi

echo $PLATFORM
echo $MINDRECORD_FILE
echo $USE_DEVICE_ID
echo $PRETRAINED_BACKBONE

echo 'start training'
export RANK_ID=0
rm -rf ${current_exec_path}/device$USE_DEVICE_ID
echo 'start device '$USE_DEVICE_ID
mkdir ${current_exec_path}/device$USE_DEVICE_ID
cd ${current_exec_path}/device$USE_DEVICE_ID  || exit
dev=`expr $USE_DEVICE_ID + 0`
export DEVICE_ID=$dev
python ${dirname_path}/${SCRIPT_NAME} \
    --run_platform=$PLATFORM \
    --mindrecord_path=$MINDRECORD_FILE \
    --pretrained=$PRETRAINED_BACKBONE > train.log  2>&1 &

echo 'running'
