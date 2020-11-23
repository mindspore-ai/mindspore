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
if [ $# != 2 ]
then
    echo "GPU: sh run_eval_for_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]"
exit 1
fi

# check dataset file
if [ ! -d $1 ]
then
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi

# check checkpoint file
if [ ! -f $2 ]
then
    echo "error: CHECKPOINT_PATH=$2 is not a file"
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

python ${BASEPATH}/../eval.py --platform 'GPU' --data_path $1 --checkpoint=$2 > ./eval.log 2>&1 &
