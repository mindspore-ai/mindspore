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

if [ $# -lt 1 ]
then
    echo "Usage: sh scripts/run_eval.sh [RANK_TABLE_FILE] --opt1 opt1_value --opt2 opt2_value ..."
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
PATH1=$(realpath $1)
export RANK_TABLE_FILE=$PATH1
echo "RANK_TABLE_FILE=${PATH1}"

export PYTHONPATH=$PWD:$PYTHONPATH
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./eval_parallel$i
    mkdir ./eval_parallel$i
    cp -r ./src ./eval_parallel$i
    cp -r ./model_utils ./eval_parallel$i
    cp -r ./*.yaml ./eval_parallel$i
    cp ./eval.py ./eval_parallel$i
    echo "start evaluation for rank $RANK_ID, device $DEVICE_ID"
    cd ./eval_parallel$i ||exit
    env > env.log
    export args=${*:2}
    python eval.py $args > eval.log 2>&1 &
    cd ..
done
