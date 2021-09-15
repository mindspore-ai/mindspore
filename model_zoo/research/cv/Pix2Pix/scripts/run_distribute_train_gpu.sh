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

if [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [DATASET_NAME] [DEVICE_NUM]"
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

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi

export DEVICE_NUM=$3
export RANK_SIZE=$3

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
mkdir ./train_parallel/results
mkdir ./train_parallel/results/ckpt
mkdir ./train_parallel/results/predict
mkdir ./train_parallel/results/loss_show
mkdir ./train_parallel/results/fake_img
cd ./train_parallel || exit

if [ $2 == 'facades' ];
then
    mpirun -n $3 --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root python train.py --device_target GPU \
    --run_distribute 1 --device_num $3 --dataset_size 400 --train_data_dir $PATH1 --pad_mode REFLECT &> log &
elif [ $2 == 'maps' ];
then
    mpirun --allow-run-as-root -n $3 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --device_target GPU --device_num $3 --dataset_size 1096 \
    --run_distribute 1 --train_data_dir $PATH1 --pad_mode REFLECT &> log &
fi
