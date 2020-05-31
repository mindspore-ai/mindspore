#!/bin/bash
# bash run_multinpu_train.sh
execute_path=$(pwd)

export RANK_TABLE_FILE=${execute_path}/rank_table_8p.json
export RANK_SIZE=8
export MINDSPORE_HCCL_CONFIG_PATH=${execute_path}/rank_table_8p.json

for((i=0;i<=7;i++));
do
  rm -rf ${execute_path}/device_$i/
  mkdir ${execute_path}/device_$i/
  cd ${execute_path}/device_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  pytest -s ${execute_path}/train_and_test_multinpu.py >train_deep$i.log 2>&1 &
done
