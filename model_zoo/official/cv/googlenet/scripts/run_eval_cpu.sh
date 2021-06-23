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

ulimit -u unlimited
# check checkpoint file
if [ ! -f $1 ]
then
    echo "error: CHECKPOINT_PATH=$1 is not a file"
exit 1
fi

dataset_type='cifar10'
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

config_path="${BASEPATH}/../${dataset_type}_config_cpu.yaml"
echo "config path is : ${config_path}"

nohup python ${BASEPATH}/../eval.py --config_path=$config_path --checkpoint_path=$1 --dataset_name=$dataset_type > ./eval.log 2>&1 &
