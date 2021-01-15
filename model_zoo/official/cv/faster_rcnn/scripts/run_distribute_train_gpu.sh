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
echo "sh run_distribute_train_gpu.sh DEVICE_NUM PRETRAINED_PATH"
echo "for example: sh run_distribute_train_gpu.sh 8 /path/pretrain.ckpt"
echo "It is better to use absolute path."
echo "=============================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_PATH]"
exit 1
fi

rm -rf run_distribute_train
mkdir run_distribute_train
cp -rf ../src/ ../train.py ./run_distribute_train
cd run_distribute_train || exit

export RANK_SIZE=$1
PRETRAINED_PATH=$2
echo "start training on $RANK_SIZE devices"

mpirun -n $RANK_SIZE \
    python train.py  \
    --run_distribute=True \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --pre_trained=$PRETRAINED_PATH  > log 2>&1 &
