#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ] && [ $# != 3 ]
then 
    echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
if [ $# == 3 ]
then 
    PATH2=$(get_real_path $3)
fi


if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi 

if [ $# == 4 ] && [ ! -f $PATH2 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
    exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8

export MS_SCHED_NUM=1
export MS_WORKER_NUM=8
export MS_SERVER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8081

export MS_ROLE=MS_SCHED
rm -rf ./sched
mkdir ./sched
cp ../config/*.yaml ./sched 
cp ../*.py ./sched
cp *.sh ./sched
cp -r ../src ./sched
cd ./sched || exit
if [ $# == 2 ]
then
    mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU" \
    --data_path=$PATH1 --parameter_server=True --config_path=$CONFIG_FILE --output_path './output' &> sched.log &
fi

if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU" \
    --data_path=$PATH1 --parameter_server=True --pre_trained=$PATH2 --config_path=$CONFIG_FILE --output_path './output' &> sched.log &
fi
cd ..

export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
    rm -rf ./server_$i
    mkdir ./server_$i
    cp ../config/*.yaml ./server_$i
    cp ../*.py ./server_$i
    cp *.sh ./server_$i
    cp -r ../src ./server_$i
    cd ./server_$i || exit
    if [ $# == 2 ]
    then
        mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
        python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU" \
        --data_path=$PATH1 --parameter_server=True --config_path=$CONFIG_FILE --output_path './output' &> server_$i.log &
    fi
        
    if [ $# == 3 ]
    then
        mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
        python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU" \
        --data_path=$PATH1 --parameter_server=True --pre_trained=$PATH2 \
        --config_path=$CONFIG_FILE --output_path './output' &> server_$i.log &
    fi
    cd ..
done

export MS_ROLE=MS_WORKER
rm -rf ./worker
mkdir ./worker
cp ../config/*.yaml ./worker 
cp ../*.py ./worker
cp *.sh ./worker
cp -r ../src ./worker
cd ./worker || exit
if [ $# == 2 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU" \
    --data_path=$PATH1 --parameter_server=True --config_path=$CONFIG_FILE --output_path './output' &> worker.log &
fi

if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --device_num=$DEVICE_NUM --device_target="GPU"\
    --data_path=$PATH1 --parameter_server=True --pre_trained=$PATH2 \
    --config_path=$CONFIG_FILE --output_path './output' &> worker.log &
fi
cd ..
