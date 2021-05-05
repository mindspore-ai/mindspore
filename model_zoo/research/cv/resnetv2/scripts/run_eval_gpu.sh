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
if [ $1 != "resnetv2_50" ] && [ $1 != "resnetv2_101" ] && [ $1 != "resnetv2_152" ]
then 
  echo "error: the selected net is neither resnetv2_50 nor resnetv2_101 and resnetv2_152"
  exit 1
fi

if [ $2 != "cifar10" ] && [ $2 != "imagenet2012" ]
then 
    echo "error: the selected dataset is neither cifar10 nor imagenet2012"
    exit 1
fi

if [ ! -d $3 ]
then 
    echo "error: DATASET_PATH=$3 is not a directory"
    exit 1
fi 

if [ ! -f $4 ]
then 
    echo "error: CHECKPOINT_PATH=$4 is not a file"
    exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

echo "start evaluation for device $DEVICE_ID"
python eval.py --net=$1 --dataset=$2 --device_target="GPU" --dataset_path=$3 --checkpoint_path=$4 &> eval.log &