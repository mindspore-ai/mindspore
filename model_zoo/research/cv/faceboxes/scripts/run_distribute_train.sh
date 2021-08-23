#!/usr/bin/env bash
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
run_ascend()
{
    if [ $# -gt 6 ] || [ $# -lt 5 ]
    then
        echo "Usage:
              Ascend: sh run_train.sh Ascend [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH]\n "
        exit 1
    fi;

    if [ $2 -lt 1 ] && [ $2 -gt 8 ]
    then
        echo "error: DEVICE_NUM=$2 is not in (1-8)"
    exit 1
    fi

    if [ ! -d $5 ] && [ ! -f $5 ]
    then
        echo "error: DATASET_PATH=$5 is not a directory or file"
    exit 1
    fi

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    VISIABLE_DEVICES=$3
    IFS="," read -r -a CANDIDATE_DEVICE <<< "$VISIABLE_DEVICES"
    if [ ${#CANDIDATE_DEVICE[@]} -ne $2 ]
    then
        echo "error: DEVICE_NUM=$2 is not equal to the length of VISIABLE_DEVICES=$3"
    exit 1
    fi
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    export RANK_TABLE_FILE=$4
    export RANK_SIZE=$2
    if [ -d "../train" ];
    then
        rm -rf ../train
    fi
    mkdir ../train
    cd ../train || exit
    for((i=0; i<${RANK_SIZE}; i++))
    do
        export DEVICE_ID=${CANDIDATE_DEVICE[i]}
        export RANK_ID=$i
        rm -rf ./rank$i
        mkdir ./rank$i
        cp ../*.py ./rank$i
        cp -r ../src ./rank$i
        cd ./rank$i || exit
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        env > env.log
        python -u ${BASEPATH}/../train.py \
            --device_target=$1 \
            --dataset_path=$5 \
            &> log$i.log & 
        cd ..
    done
}

if [ $1 = "Ascend" ] ; then
    run_ascend "$@"
else
    echo "Unsupported device_target"
fi;
