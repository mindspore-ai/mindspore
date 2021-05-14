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

ulimit -u unlimited

if [ $# != 1 ] && [ $# != 2 ]
then
    echo "GPU: sh run_eval_gpu.sh [CHECKPOINT_PATH] [cifar10|imagenet]"
exit 1
fi

# check checkpoint file
if [ ! -f $1 ]
then
    echo "error: CHECKPOINT_PATH=$1 is not a file"    
exit 1
fi

dataset_type='cifar10'
if [ $# == 2 ]
then
    if [ $2 != "cifar10" ] && [ $2 != "imagenet" ]
    then
        echo "error: the selected dataset is neither cifar10 nor imagenet"
    exit 1
    fi
    dataset_type=$2
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
export DEVICE_ID=0

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

config_path="${BASEPATH}/../${dataset_type}_config.yaml"
echo "config path is : ${config_path}"

python3 ${BASEPATH}/../eval.py --config_path=$config_path --checkpoint_path=$1 --dataset_name=$dataset_type > ./eval.log 2>&1 &
