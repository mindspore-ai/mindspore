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

if [ $# != 5 ]
then
    echo "Usage: sh scripts/run_eval.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [checkpoint_path] [instances_set]"
exit 1
fi

DATASET=$2
MINDRECORD_DIR=$3
CHECKPOINT_PATH=$4
INSTANCE_SET=$5
echo $DATASET

export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit

if [ -d "eval$2" ];
then
    rm -rf ./eval$2
fi

mkdir ./eval$2
cp ./*.py ./eval$2
cp -r ./src ./eval$2
cp ./*yaml ./eval$2
cd ./eval$2 || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval.py \
    --dataset=$DATASET \
    --checkpoint_path=$CHECKPOINT_PATH \
    --instances_set=$INSTANCE_SET \
    --mindrecord_dir=$MINDRECORD_DIR > log.txt 2>&1 &
cd ..
