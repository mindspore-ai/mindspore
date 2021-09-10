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
    echo "sh convert_pretrain_dataset.sh [DATASET_PATH] [OUTPUT_PATH]"
    echo "for example: sh scripts/convert_pretrain_dataset.sh /path/zh_wiki/ /path/zh_wiki/mindrecord/"
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

CUR_DIR=`pwd`
MODEL_PATH=${CUR_DIR}/pretrain_models/converted

# ner task
# train dataset
python ${CUR_DIR}/src/pretrain_reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=512 \
    --random_seed=1 \
    --do_lower_case="true" \
    --short_seq_prob=0.1 \
    --masked_word_prob=0.15 \
    --max_predictions_per_seq=20 \
    --dupe_factor=10 \
    --generate_neg_sample="true" \
    --input_file=$DATASET_PATH \
    --output_file=$OUTPUT_PATH/pretrain_data.mindrecord \
    --shard_num=10