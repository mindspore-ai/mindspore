#!/usr/bin/env bash
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
if [ $# != 1 ] && [ $# != 2 ]  && [ $# != 3 ]
then
  echo "Usage bash scripts/run_train_cpu.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]"
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
  echo "error: VAL_DATA_DIR=$PATH1 is not a directory"
exit 1
fi

PATH2=$(get_real_path $3)
if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

BASE_PATH=$(dirname "$(dirname "$(readlink -f $0)")")
if [ $2 == 'imagenet' ]; then
  CONFIG_FILE="${BASE_PATH}/config/imagenet_config.yaml"
elif [ $2 == 'cifar10' ]; then
  CONFIG_FILE="${BASE_PATH}/config/cifar10_config.yaml"
else
  echo "error: the selected dataset is neither cifar10 nor imagenet"
exit 1
fi

rm -rf ./eval
mkdir ./eval
cp -r ./src ./eval
cp ./eval.py ./eval
cp -r ./config ./eval
env >env.log
echo "start evaluation for device CPU"
cd ./eval || exit
python ./eval.py --device_target=CPU --val_data_dir=$PATH1 --dataset_name=$2 --config_path=$CONFIG_FILE \
--checkpoint_path=$PATH2 > ./eval.log 2>&1 &
cd ..
