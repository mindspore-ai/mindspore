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


get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 3 ]  && [ $# != 4 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_distribute_train_gpu.sh [RANKSIZE] [DATASET] [CONFIG_PATH] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)](optional)"
  echo "for example: bash run_distribute_train_gpu.sh 8 /path/to/data/ /path/to/config/"
  echo "=============================================================================================================="
  exit 1
fi

RANK_SIZE=`expr $1 + 0`
if [ $? != 0 ]; then
  echo RANK_SIZE=$1 is not integer!
  exit 1
fi
export RANK_SIZE=$RANK_SIZE
DATASET=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)
if [ $# != 4 ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
  export CUDA_VISIBLE_DEVICES=$4
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
TRAIN_OUTPUT=${PROJECT_DIR}/../train_distributed_gpu
if [ -d $TRAIN_OUTPUT ]; then
  rm -rf $TRAIN_OUTPUT
fi
mkdir $TRAIN_OUTPUT
cd $TRAIN_OUTPUT || exit
cp ../train.py ./
cp ../eval.py ./
cp -r ../src ./
cp $CONFIG_PATH ./
env > env.log

mpirun -n $RANK_SIZE --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python train.py  --run_distribute=True \
                 --data_path=$DATASET  \
                 --config_path=${CONFIG_PATH##*/}  \
                 --output=./output \
                 --device_target=GPU> train.log 2>&1 &
