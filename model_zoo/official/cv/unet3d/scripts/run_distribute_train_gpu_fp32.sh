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

if [ $# -ne 1 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DATA_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1
if [ ! -d $PATH1 ]
then
    echo "error: IMAGE_PATH=$PATH1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8


if [ -d "train_parallel_fp32" ];
then
    rm -rf ./train_parallel_fp32
fi

rm -rf ./train_parallel_fp32
mkdir ./train_parallel_fp32
cp ../*.py ./train_parallel_fp32
cp *.sh ./train_parallel_fp32
cp ../*.yaml ./train_parallel_fp32
cp -r ../src ./train_parallel_fp32
cd ./train_parallel_fp32 || exit
echo "start distributed training with $DEVICE_NUM GPUs."
env > env.log
mpirun --allow-run-as-root -n $DEVICE_NUM python train.py --run_distribute=True --data_path=$PATH1 --output_path './output' --device_target='GPU' --checkpoint_path='./' > train.log 2>&1 &
cd ..
