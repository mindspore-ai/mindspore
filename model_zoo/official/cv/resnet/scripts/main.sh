#!/bin/bash

ulimit -u unlimited

# 单机单卡
if [ $# == 1 ]; then
    export DEVICE_NUM=1
    export DEVICE_ID=0
    export RANK_ID=0
    export RANK_SIZE=1
    unset RANK_TABLE_FILE

    if [ -d "train" ];
    then
        rm -rf ./train
    fi
    mkdir ./train
    cp ../*.py ./train
    cp *.sh ./train
    cp -r ../src ./train
    cd ./train || exit
    echo "start training for device $DEVICE_ID"
    env > env.log

    # 保持前台输出
    python train.py --net=resnet50 --dataset=cifar10 --dataset_path=$1 | tee log
fi

# 单机多卡和分布式
if [ $# == 5 ]; then
    export DEVICE_NUM=$1
    export SERVER_NUM=$2
    export RANK_SIZE=$1
    export RANK_TABLE_FILE=$3

    export SERVER_ID=$4
    device_each_server=$((DEVICE_NUM / SERVER_NUM))
    rank_start=$((${device_each_server} * SERVER_ID))

    # 先启动后台任务，最后留一个前台任务查看日志输出
    for((i=$(($device_each_server-1)); i>=0; i--))
    do
        rankid=$((rank_start + i))
        export DEVICE_ID=${i}
        export RANK_ID=${rankid}
        rm -rf ./train_parallel${rankid}
        mkdir ./train_parallel${rankid}
        cp ../*.py ./train_parallel${rankid}
        cp *.sh ./train_parallel${rankid}
        cp -r ../src ./train_parallel${rankid}
        cd ./train_parallel${rankid} || exit
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        env > env.log

        if [ $i -eq 0 ]; then
            python train.py --net=resnet50 --dataset=cifar10 --run_distribute=True --device_num=$device_each_server --dataset_path=$5 | tee log
        else
            python train.py --net=resnet50 --dataset=cifar10 --run_distribute=True --device_num=$device_each_server --dataset_path=$5 &> log &
        fi
    done
else
    echo "Invalid input parameter, usage: main.sh device_count server_count RANK_TABLE_FILE server_id dataset" | tee log
    exit 1
fi

wait
