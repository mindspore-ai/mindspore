#!/usr/bin/env bash
if [ $# != 4 ]
then
    echo "Usage: sh run_train.sh [DEVICE_NUM] [SERVER_IP(x.x.x.x)] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]"
exit 1
fi

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi

if [ ! -d $4 ]
then
    echo "error: DATASET_PATH=$4 is not a directory"
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cd ./train || exit
python ${BASEPATH}/launch.py \
        --nproc_per_node=$1 \
        --visible_devices=$3 \
        --server_id=$2 \
        --training_script=${BASEPATH}/train.py \
        --dataset_path=$4 &> train.log &  # dataset train folder
