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


get_gpu_device_num(){

    #device_list=(${1//,/ })
    IFS=',' read -ra device_list <<<"$1"
    device_used=(0 0 0 0 0 0 0 0)
    device_num=0
    for var in "${device_list[@]}"
    do  
        if [ $((var)) -lt 0 ] || [ $((var)) -ge 8 ]
        then 
            echo "error: device id=${var} is incorrect, device id must be in range [0,8), please check your device id list!"
            exit 1
        fi

        if [  ${device_used[$((var))]} -eq 0 ]
        then 
            device_used[ $((var)) ]=1
            device_num=$((device_num+1))
        fi
    done

    echo ${device_num}
}


run_ascend(){

    if [ $# -gt 4 ] || [ $# -lt 3 ]
    then
        echo "Usage:  bash run_train.sh [Ascend] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)\n "
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



    #rank_file_name=${2##*/}
    #IFS='_' read -ra array <<<"${rank_file_name}"
    #device_id_list=${array[2]}
    #first_device=${device_id_list:0:1}
    #last_device=${device_list:${#device_list}-1:1}
    #device_num=${#device_id_list}
    cat $2 | awk -F "[device_id]" '/device_id/{print$0}' >temp.log
    array=$(cat temp.log | awk -F "[:]" '/device_id/{print$2}')
    
    IFS=" " read -ra device_list <<<$array
    first_device=${device_list[0]:1:1}
    #device_num=${#device_list[*]}
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
        cd ./train/device$i || exit
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        env > env.log
        if [ $# == 3 ]
        then    
            python train.py --device_target=$1  --dataset_path=$PATH2 &> train.log &
        fi
        
        if [ $# == 4 ]
        then
            python train.py --device_target=$1  --dataset_path=$PATH2 --pre_trained=$PATH3 &> train.log &
        fi

        cd ../.. || exit
    done
}

run_gpu(){
    if [ $# -gt 4 ] || [ $# -lt 3 ]
    then
        echo "Usage:  bash run_train.sh  [GPU] [DEVICE_ID_LIST] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)\n "
        exit 1
    fi

    PATH1=$(get_real_path $3)
    
    if [ $# == 4 ]
    then 
        PATH2=$(get_real_path $4)
    fi

    if [ ! -d $PATH1 ]
    then 
        echo "error: DATASET_PATH=$PATH1 is not a directory"
        exit 1
    fi 

    if [ $# == 4 ] && [ ! -f $PATH2 ]
    then
        echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
        exit 1
    fi

    device_num=$(get_gpu_device_num $2)

    ulimit -u unlimited
    export DEVICE_NUM=${device_num}
    export RANK_SIZE=${device_num}
    export CUDA_VISIBLE_DEVICES=$2
    
    rm -rf ./train
    mkdir ./train
    cp ../*.py ./train
    cp *.sh ./train
    cp -r ../src ./train
    cd ./train || exit
    echo "start training"
    env > env.log
    if [ $# == 3 ]
    then
        mpirun --allow-run-as-root -n ${RANK_SIZE} --output-filename log_output --merge-stderr-to-stdout \
        python train.py --device_target=$1  --dataset_path=$PATH1 &> train.log &
    fi
    
    if [ $# == 4 ]
    then
        mpirun --allow-run-as-root -n ${RANK_SIZE} --output-filename log_output --merge-stderr-to-stdout \
        python train.py --device_target=$1  --dataset_path=$PATH1 --pre_trained=$PATH2 &> train.log &
    fi

    cd ..
}


if [ $1 = "Ascend" ] ; then
    run_ascend "$@"
elif [ $1 = "GPU" ] ; then
    run_gpu "$@"
else
    echo "Unsupported device target: $1"
fi;

