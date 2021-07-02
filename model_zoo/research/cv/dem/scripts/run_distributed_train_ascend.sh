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
ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=600
export RANK_TABLE_FILE=$1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

# remove old train_parallel files
rm -rf ../train_parallel
mkdir ../train_parallel

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

echo "device num=$DEVICE_NUM"

DATASET=$2
TRAIN_MODE=$3
DATA_PATH=$4

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    # mkdirs
    mkdir ../train_parallel/$i
    mkdir ../train_parallel/$i/src
    # move files
    cp ../*.py ../train_parallel/$i
    cp ../src/*.py ../train_parallel/$i/src
    
    # goto the training dirs of each training
    cd ../train_parallel/$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    # input logs to env.log
    env > env.log
    # start training single task
    python train.py --device_id=$i \
        --distribute=True \
        --dataset=$DATASET \
        --train_mode=$TRAIN_MODE \
        --data_path=$DATA_PATH  &> log_train_8p.txt &
    cd ../../scripts
done
