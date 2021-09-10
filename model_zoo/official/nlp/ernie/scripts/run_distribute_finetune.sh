#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
if [ $# -ne 2 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh run_distribute_finetune.sh [RANK_TABLE_FILE] [TASK_TYPE]"
    echo "for example: sh scripts/run_distribute_finetune.sh rank_table.json xnli"
    echo "TASK_TYPE including [xnli, dbqa, drcd, cmrc]"
    echo "=============================================================================================================="
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
echo $PATH1

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

ulimit -u unlimited
mkdir -p ms_log
mkdir -p save_models
CUR_DIR=`pwd`
MODEL_PATH=${CUR_DIR}/pretrain_models/converted
DATA_PATH=${CUR_DIR}/data
SAVE_PATH=${CUR_DIR}/save_models
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export RANK_TABLE_FILE=$PATH1
DEVICE_NUM=8
START_DEVICE_NUM=0

TASK_TYPE=$2
case $TASK_TYPE in
  "xnli")
    PY_NAME=run_ernie_classifier
    NUM_LABELS=3
    NUM_EPOCH=3
    TRAIN_BATCH_SIZE=64
    EVAL_BATCH_SIZE=64
    TRAIN_DATA_PATH="${DATA_PATH}/xnli/xnli_train.mindrecord0"
    EVAL_DATA_PATH="${DATA_PATH}/xnli/xnli_dev.mindrecord"
    EVAL_JSON_PATH=""
    ;;
  "dbqa")
    PY_NAME=run_ernie_classifier
    NUM_LABELS=2
    NUM_EPOCH=3
    TRAIN_BATCH_SIZE=64
    EVAL_BATCH_SIZE=64
    TRAIN_DATA_PATH="${DATA_PATH}/nlpcc-dbqa/dbqa_train.mindrecord0"
    EVAL_DATA_PATH="${DATA_PATH}/nlpcc-dbqa/dbqa_dev.mindrecord"    
    EVAL_JSON_PATH="${DATA_PATH}/nlpcc-dbqa/dev.json"
    ;;
  "drcd")
    PY_NAME=run_ernie_mrc
    NUM_LABELS=2
    NUM_EPOCH=3
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=16
    TRAIN_DATA_PATH="${DATA_PATH}/drcd/drcd_train.mindrecord0"
    EVAL_DATA_PATH="${DATA_PATH}/drcd/drcd_dev.mindrecord"
    EVAL_JSON_PATH="${DATA_PATH}/drcd/dev.json"  
    ;;
  "cmrc")
    PY_NAME=run_ernie_mrc
    NUM_LABELS=2
    NUM_EPOCH=3
    TRAIN_BATCH_SIZE=16
    EVAL_BATCH_SIZE=16
    TRAIN_DATA_PATH="${DATA_PATH}/cmrc2018/cmrc_train.mindrecord"
    EVAL_DATA_PATH="${DATA_PATH}/cmrc2018/cmrc_dev.mindrecord"
    EVAL_JSON_PATH="${DATA_PATH}/cmrc2018/dev.json"  
    ;;
  esac

for((i=0; i<$DEVICE_NUM; i++))
do
    export DEVICE_ID=`expr $i + $START_DEVICE_NUM`
    export RANK_ID=$i
    if [ $i -eq 0 ]
    then
        DO_EVAL="true"
    else
        DO_EVAL="false"
    fi
    python ${CUR_DIR}/$PY_NAME.py \
        --task_type=$TASK_TYPE \
        --device_target="Ascend" \
        --run_distribute="true" \
        --vocab_path="${MODEL_PATH}/vocab.txt" \
        --do_train="true" \
        --do_eval=$DO_EVAL \
        --device_num=$DEVICE_NUM \
        --device_id=$DEVICE_ID \
        --rank_id=$i \
        --epoch_num=$NUM_EPOCH \
        --number_labels=$NUM_LABELS \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --train_batch_size=$TRAIN_BATCH_SIZE \
        --eval_batch_size=$EVAL_BATCH_SIZE \
        --save_finetune_checkpoint_path="${SAVE_PATH}" \
        --load_pretrain_checkpoint_path="${MODEL_PATH}/ernie.ckpt" \
        --train_data_file_path=$TRAIN_DATA_PATH \
        --eval_data_file_path=$EVAL_DATA_PATH \
        --eval_json_path=$EVAL_JSON_PATH > ${GLOG_log_dir}/${TASK_TYPE}_train_log_$i.txt 2>&1 &
done