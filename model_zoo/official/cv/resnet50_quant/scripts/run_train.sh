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

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

run_ascend(){

    if [ $# != 3 ] && [ $# != 4 ]
    then 
        echo "Usage: bash run_train.sh [Ascend] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)\n"
        exit 1
    fi

    PATH1=$(get_real_path $2)
    PATH2=$(get_real_path $3)

    if [ $# == 4 ]
    then 
        PATH3=$(get_real_path $4)
    fi

    if [ ! -f $PATH1 ]
    then 
        echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
        exit 1
    fi 

    if [ ! -d $PATH2 ]
    then 
        echo "error: DATASET_PATH=$PATH2 is not a directory"
        exit 1
    fi 

    if [ $# == 4 ] && [ ! -f $PATH3 ]
    then
        echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
        exit 1
    fi

    cat $2 | grep device_id >temp.log
    array=$(cat temp.log | awk -F "[:]" '{print$2}')
    IFS=" " read -ra device_list <<<$array
    first_device=${device_list[0]:1:1}
    device_num=$(cat temp.log | wc -l)
    rm temp.log

    ulimit -u unlimited
    export DEVICE_NUM=${device_num}
    export RANK_SIZE=${device_num}
    export RANK_TABLE_FILE=$PATH1

    export SERVER_ID=0
    rank_start=$((DEVICE_NUM * SERVER_ID))

    rm -rf ./train
    mkdir ./train
    for((i=0; i<${device_num}; i++))
    do
        export DEVICE_ID=$((first_device+i))
        export RANK_ID=$((rank_start + i))
        mkdir ./train/device$i
        cp ../*.py ./train/device$i
        cp *.sh ./train/device$i
        cp -r ../src ./train/device$i
        cp -r ../models ./train/device$i
        cd ./train/device$i || exit
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        env > env.log
        if [ $# == 3 ]
        then
            python train.py  --device_target=$1 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 &> train.log &
        fi
        
        if [ $# == 4 ]
        then
            python train.py  --device_target=$1 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 --pre_trained=$PATH3 &> train.log &
        fi

        cd ../.. || exit
    done    
}

# run_gpu(){

#     if [ $# -gt 3 ] || [ $# -lt 2 ]
#     then
#         echo "Usage:  sh run_train_distribute_quant.sh  [GPU] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)\n "
#         exit 1
#     fi

#     PATH1=$(get_real_path $2)
    
#     if [ $# == 3 ]
#     then 
#         PATH2=$(get_real_path $3)
#     fi

#     if [ ! -d $PATH1 ]
#     then 
#         echo "error: DATASET_PATH=$PATH1 is not a directory"
#         exit 1
#     fi 

#     if [ $# == 3 ] && [ ! -f $PATH2 ]
#     then
#         echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
#         exit 1
#     fi

#     ulimit -u unlimited
#     export RANK_SIZE=2
#     #export CUDA_VISIBLE_DEVICES=1,2

#     rm -rf ./train_parallel
#     mkdir ./train_parallel
#     cp ../*.py ./train_parallel
#     cp *.sh ./train_parallel
#     cp -r ../src ./train_parallel
#     cp -r ../models ./train_parallel
#     cd ./train_parallel || exit
#     echo "start training"
#     env > env.log
#     if [ $# == 2 ]
#     then
#         mpirun --allow-run-as-root -n $RANK_SIZE
#         python train.py --device_target=$1  --dataset_path=$PATH1 &> log &
#     fi
    
#     if [ $# == 3 ]
#     then
#         mpirun --allow-run-as-root -n $RANK_SIZE
#         python train.py --device_traget=$1  --dataset_path=$PATH1 --pre_trained=$PATH2 &> log &
#     fi
#     cd ..
# }


if [ $1 = "Ascend" ] ; then
    run_ascend "$@"
else
    echo "Unsupported device target: $1"
fi;
