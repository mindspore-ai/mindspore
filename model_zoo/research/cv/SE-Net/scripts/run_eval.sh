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

if [ $# != 2 ]
then 
    echo "Usage: sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0
export DATA_PATH=$1
export CKPT_PATH=$2

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"

#python eval.py --dataset_path /data/imagenet/val/ --checkpoint_path /home/wks/train8s/train_paralleltrain_parallel0/resnet-90_625.ckpt | tee eval.log # export ID=0
python eval.py --dataset_path=$DATA_PATH --checkpoint_path=$CKPT_PATH  &> log &

cd ..
