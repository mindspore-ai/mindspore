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

if [ $# -lt 2 ]
then
    echo "Usage: sh run_train_gpu.sh [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]"
exit 1
fi

export DEVICE_NUM=1
DATASET_PATH=$2

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train

if [ -d "ckpts" ];
then
    rm -rf ./ckpts
fi
mkdir ./ckpts

cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
env > env.log
echo "start training"

export CUDA_VISIBLE_DEVICES="$1"

python train.py --datapath=$DATASET_PATH --ckptpath=../ckpts \
                --device_target='GPU' --num_epoch=680 \
                --dist_reg=0 > log 2>&1 &

cd ..
