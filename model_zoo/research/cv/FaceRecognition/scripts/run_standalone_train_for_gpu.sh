#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

if [ $# != 2 ]  && [ $# != 1 ]; then
  echo "Usage: bash run_standalone_train_for_gpu.sh [base/beta] [DEVICE_ID](optional)"
  exit 1
fi

expr $2 + 6 &>/dev/null
if [ $? != 0 ]
then
    echo "error:DEVICE_ID=$2 is not a integer"
exit 1
fi

if [ $# -eq 2 ]; then
  DEVICE_ID=$2
else
  DEVICE_ID=0
fi

if [ $1 = "base" ]; then
  CONFIG_PATH='./base_config.yaml'
elif [ $1 = "beta" ]; then
  CONFIG_PATH='./beta_config.yaml'
else
  echo "error: the train_stage is neither base nor beta"
exit 1
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export DEVICE_ID=0
TRAIN_STAGE=$1
TRAIN_OUTPUT=./standalone_train_for_gpu_$TRAIN_STAGE
EXECUTE_PATH=$(pwd)
echo *******************EXECUTE_PATH=$EXECUTE_PATH
echo TRAIN_OUTPUT=$TRAIN_OUTPUT
echo CONFIG_PATH=$CONFIG_PATH
echo TRAIN_STAGE=$TRAIN_STAGE
echo '*********************************************'
if [ -d $TRAIN_OUTPUT ]; then
  rm -rf $TRAIN_OUTPUT
fi

mkdir $TRAIN_OUTPUT
cp ../train.py $TRAIN_OUTPUT
cp ../*.yaml $TRAIN_OUTPUT
cp -r ../model_utils $TRAIN_OUTPUT
cp -r ../src $TRAIN_OUTPUT
cd $TRAIN_OUTPUT || exit

python train.py  \
  --config_path=$CONFIG_PATH \
  --device_target=GPU \
  --train_stage=$TRAIN_STAGE \
  --is_distributed=0 &> train_standalone_for_gpu.log &
cd ..