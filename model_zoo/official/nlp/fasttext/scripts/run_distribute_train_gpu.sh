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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_distributed_train_gpu.sh DATASET_PATH DEVICE_NUM"
echo "for example: sh run_distributed_train_gpu.sh /home/workspace/ag 8"
echo "It is better to use absolute path."
echo "=============================================================================================================="
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
echo $DATASET
DATANAME=$(basename $DATASET)

echo $DATANAME


if [ -d "distribute_train" ];
then
    rm -rf ./distribute_train
fi
mkdir ./distribute_train
cp ../*.py ./distribute_train
cp -r ../src ./distribute_train
cp -r ../scripts/*.sh ./distribute_train
cd ./distribute_train || exit
echo "start training for $2 GPU devices"

mpirun -n $2 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python ../../train.py --device_target GPU --run_distribute True --data_path $DATASET --data_name $DATANAME
cd ..
