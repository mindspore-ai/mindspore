#!/bin/bash

export DEVICE_NUM=8
export RANK_SIZE=8
export MS_ENABLE_GE=1

file_name=$1
cache_path=$2
log_name=$3
rank_table_path=$4
export RANK_TABLE_FILE=$rank_table_path
export GLOG_v=2
 

for ((i=0; i<8; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    log_fullname=$log_name$i
    python $file_name $cache_path &> $log_fullname &
done
