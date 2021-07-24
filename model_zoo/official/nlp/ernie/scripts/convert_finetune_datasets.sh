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
if [ $# -ne 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh convert_finetune_dataset.sh [DATASET_PATH] [OUTPUT_PATH] [TASK_TYPE]"
    echo "for example: sh scripts/convert_finetune_dataset.sh /path/msra_ner/ /path/msra_ner/mindrecord/ msra_ner"
    echo "TASK_TYPE including [msra_ner, chnsenticorp, xnli, dbqa, drcd, cmrc]"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
exit 1
fi

ulimit -u unlimited
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATASET_PATH=$(get_real_path $1)
echo $DATASET_PATH
if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not valid"
exit 1
fi
OUTPUT_PATH=$(get_real_path $2)
echo $OUTPUT_PATH
if [ ! -d $OUTPUT_PATH ]
then
    echo "error: OUTPUT_PATH=$OUTPUT_PATH is not valid"
exit 1
fi

TASK_TYPE=$3
case $TASK_TYPE in
  "msra_ner")
    MAX_SEQ_LEN=256
    MAX_QUERY_LEN=0
    SHARD_NUM=1
    FILE_TYPE="tsv"
    HAVE_TEST="true"
    ;;
  "chnsenticorp")
    MAX_SEQ_LEN=256
    MAX_QUERY_LEN=0
    SHARD_NUM=1
    FILE_TYPE="tsv"
    HAVE_TEST="true"
    ;;
  "xnli")
    MAX_SEQ_LEN=512
    MAX_QUERY_LEN=0
    SHARD_NUM=10
    FILE_TYPE="tsv"
    HAVE_TEST="true"
    ;;
  "dbqa")
    MAX_SEQ_LEN=512
    MAX_QUERY_LEN=0
    SHARD_NUM=10
    FILE_TYPE="tsv"
    HAVE_TEST="true"
    ;;
  "drcd")
    MAX_SEQ_LEN=512
    MAX_QUERY_LEN=64
    SHARD_NUM=10
    FILE_TYPE="json"
    HAVE_TEST="true"
    ;;
  "cmrc")
    MAX_SEQ_LEN=512
    MAX_QUERY_LEN=64
    SHARD_NUM=1
    FILE_TYPE="json"
    HAVE_TEST="false"
    ;;
  esac

CUR_DIR=`pwd`
MODEL_PATH=${CUR_DIR}/pretrain_models/ernie

# ner task
# train dataset
python ${CUR_DIR}/src/finetune_task_reader.py  \
    --task_type=$TASK_TYPE \
    --label_map_config="${DATASET_PATH}/label_map.json" \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=$MAX_SEQ_LEN \
    --max_query_len=$MAX_QUERY_LEN \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATASET_PATH}/train.${FILE_TYPE}" \
    --output_file="${OUTPUT_PATH}/${TASK_TYPE}_train.mindrecord" \
    --shard_num=$SHARD_NUM \
    --is_training="true"

# dev dataset
python ${CUR_DIR}/src/finetune_task_reader.py  \
    --task_type=$TASK_TYPE \
    --label_map_config="${DATASET_PATH}/label_map.json" \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=$MAX_SEQ_LEN \
    --max_query_len=$MAX_QUERY_LEN \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATASET_PATH}/dev.${FILE_TYPE}" \
    --output_file="${OUTPUT_PATH}/${TASK_TYPE}_dev.mindrecord" \
    --shard_num=1 \
    --is_training="false"

# test dataset
if [ ${HAVE_TEST} = "true" ]
then
  python ${CUR_DIR}/src/finetune_task_reader.py  \
      --task_type=$TASK_TYPE \
      --label_map_config="${DATASET_PATH}/label_map.json" \
      --vocab_path="${MODEL_PATH}/vocab.txt" \
      --max_seq_len=$MAX_SEQ_LEN \
      --max_query_len=$MAX_QUERY_LEN \
      --do_lower_case="true" \
      --random_seed=1 \
      --input_file="${DATASET_PATH}/test.${FILE_TYPE}" \
      --output_file="${OUTPUT_PATH}/${TASK_TYPE}_test.mindrecord" \
      --shard_num=1 \
      --is_training="false"
fi
