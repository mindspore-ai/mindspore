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

if [ $# != 2 ] && [ $# != 3 ]; then
  echo "Usage: bash run_distribute_train_for_gpu.sh [RANK_SIZE] [base/beta] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)](optional)"
  exit 1
fi

expr $1 + 0 &>/dev/null
if [ $? != 0 ]
then
    echo "error:RANK_SIZE=$1 is not a integer"
exit 1
fi

if [ $2 = "base" ]; then
  CONFIG_PATH='./base_config.yaml'
elif [ $2 = "beta" ]; then
  CONFIG_PATH='./beta_config.yaml'
else
  echo "error: the train_stage is neither base nor beta"
exit 1
fi

if [ $# != 3 ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
  export CUDA_VISIBLE_DEVICES=$3
fi

RANK_SIZE=$1
TRAIN_STAGE=$2
TRAIN_OUTPUT=./distribute_train_for_gpu_$TRAIN_STAGE
EXECUTE_PATH=$(pwd)
echo *******************EXECUTE_PATH=$EXECUTE_PATH
echo TRAIN_OUTPUT=$TRAIN_OUTPUT
echo CONFIG_PATH=$CONFIG_PATH
echo RANK_SIZE=$RANK_SIZE
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

env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python train.py  \
    --config_path=$CONFIG_PATH \
    --device_target=GPU \
    --train_stage=$TRAIN_STAGE \
    --is_distributed=1 &> train_distribute_for_gpu.log &
cd ..