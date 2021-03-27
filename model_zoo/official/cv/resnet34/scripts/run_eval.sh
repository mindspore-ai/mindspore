#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval.sh DATA_PATH CHECKPOINT_PATH "
echo "For example: bash run.sh /path/dataset Resnet152-140_5004.ckpt"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
      else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval
env > env.log
echo "start evaluation for device $DEVICE_ID"
python eval.py --data_url=$PATH1 --checkpoint_path=$PATH2 &> eval.log &
cd ..
