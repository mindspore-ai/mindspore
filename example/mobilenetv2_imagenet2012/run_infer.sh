#!/usr/bin/env bash
if [ $# != 2 ]
then
    echo "Usage: sh run_infer.sh [DATASET_PATH] [CHECKPOINT_PATH]"
exit 1
fi

if [ ! -d $1 ]
then
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi

if [ ! -f $2 ]
then
    echo "error: CHECKPOINT_PATH=$2 is not a file"
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1
if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cd ./eval || exit
python ${BASEPATH}/eval.py \
        --checkpoint_path=$2 \
        --dataset_path=$1 &> infer.log &  # dataset val folder path
