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
if [ $# != 4 ]
then
    echo "Usage: sh run_eval.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]"
    exit 1
fi

export CKPT=$1
export EVALLRPATH=$2
export EVALGTPATH=$3
export DEVICE_ID=$4

rm -rf ./eval
mkdir ./eval
cp -r ../src ./eval
cp -r ../*.py ./eval
cd ./eval || exit

env > env.log
python ./eval.py --generator_path=$CKPT --test_LR_path=$EVALLRPATH --device_id $DEVICE_ID\
                 --test_GT_path=$EVALGTPATH &> log &
