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

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
export DEVICE_ID=0

dataset_type='cifar10'
if [ $# == 1 ]
then
    if [ $1 != "cifar10" ] && [ $1 != "imagenet" ]
    then
        echo "error: the selected dataset is neither cifar10 nor imagenet"
    exit 1
    fi
    dataset_type=$1
fi
config_path="${BASEPATH}/../${dataset_type}_config.yaml"
echo "config path is : ${config_path}"

python ${BASEPATH}/../eval.py --config_path=$config_path --dataset_name=$dataset_type > ./eval.log 2>&1 &
