#!/bin/bash
execute_path=$(pwd)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export RANK_SIZE=8
export RANK_TABLE_FILE=${execute_path}/../serving_increment/hccl_8p.json
export MODE=13B
export STRATEGY=$1
export CKPT_PATH=$2
export CKPT_NAME=$3
export PARAM_INIT_TYPE=$4

for((i=0;i<$RANK_SIZE;i++));
do
  rm -rf ${execute_path}/device_$i/
  mkdir ${execute_path}/device_$i/
  cd ${execute_path}/device_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --load_ckpt_path=$CKPT_PATH \
                  --load_ckpt_name=$CKPT_NAME --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE \
                  --export=1 >log$i.log 2>&1 &
done