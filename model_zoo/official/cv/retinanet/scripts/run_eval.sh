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
    echo "Usage: sh run_eval.sh [DATASET] [DEVICE_ID]"
exit 1
fi

DATASET=$1
echo $DATASET


export DEVICE_NUM=1
export DEVICE_ID=$2
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
cd ./eval$2 || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval.py \
    --dataset=$DATASET \
    --device_id=$2 > log.txt 2>&1 &
cd ..
