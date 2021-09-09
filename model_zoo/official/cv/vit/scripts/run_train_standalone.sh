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

if [ $# != 1 ]
then
    echo "Usage: bash run_standalone_train.sh [CONFIG_PATH] "
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_FILE=$(get_real_path $1)

if [ ! -f $CONFIG_FILE ]
then
    echo "error: config_path=$CONFIG_FILE is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../config/*.yml ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
if [ $# == 1 ]
then
    python train.py --config_path=$CONFIG_FILE &> log &
fi

cd ..