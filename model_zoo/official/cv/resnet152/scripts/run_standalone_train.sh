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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train.sh DATA_PATH PRETRAINED_CKPT_PATH(optional)"
echo "For example: bash run_standalone_train.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
if [ $# == 2 ]
then
    PATH2=$(get_real_path $2)
fi

if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ $# == 2 ] && [ ! -f $PATH2 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=6
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
if [ $# == 1 ]
then
    python train.py  --data_url=$PATH1 &> train.log &
fi

if [ $# == 2 ]
then
    python train.py  --data_url=$PATH1 --pre_trained=$PATH2 &> train.log &
fi
cd ..


