#!/usr/bin/env bash
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

run_ascend()
{
    if [ $2 -lt 1 ] && [ $2 -gt 8 ]
    then
        echo "error: DEVICE_NUM=$2 is not in (1-8)"
    exit 1
    fi

    if [ ! -d $5 ]
    then
        echo "error: DATASET_PATH=$5 is not a directory"
    exit 1
    fi

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    if [ -d "../train" ];
    then
        rm -rf ../train
    fi
    mkdir ../train
    cd ../train || exit
    python ${BASEPATH}/../src/launch.py \
            --nproc_per_node=$2 \
            --visible_devices=$4 \
            --server_id=$3 \
            --training_script=${BASEPATH}/../train.py \
            --dataset_path=$5 \
            --pre_trained=$6 \
            --device_target=$1 &> train.log &  # dataset train folder
}

run_gpu()
{
    if [ $2 -lt 1 ] && [ $2 -gt 8 ]
    then
        echo "error: DEVICE_NUM=$2 is not in (1-8)"
    exit 1
    fi

    if [ ! -d $4 ]
    then
        echo "error: DATASET_PATH=$4 is not a directory"
    exit 1
    fi

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    if [ -d "../train" ];
    then
        rm -rf ../train
    fi
    mkdir ../train
    cd ../train || exit

    export CUDA_VISIBLE_DEVICES="$3"
    mpirun -n $2 --allow-run-as-root \
    python ${BASEPATH}/../train.py \
        --dataset_path=$4 \
        --platform=$1 \
        --pre_trained=$5 \
        &> train.log &  # dataset train folder
}

if [ $# -gt 6 ] || [ $# -lt 4 ]
then
    echo "Usage:\n \
          Ascend: sh run_train.sh Ascend [DEVICE_NUM] [SERVER_IP(x.x.x.x)] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH]\n \
          GPU: sh run_train.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH]\n \
          "
exit 1
fi

if [ $1 = "Ascend" ] ; then
    run_ascend "$@"
elif [ $1 = "GPU" ] ; then
    run_gpu "$@"
else
    echo "not support platform"
fi;

