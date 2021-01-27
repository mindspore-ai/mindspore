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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh create_dataset.sh DATASET_PATH OUTPUT_PATH"
echo "for example: sh create_dataset.sh /path/multi30k/ /path/multi30k/mindrecord/"
echo "DATASET_NAME including ag, dbpedia, and yelp_p"
echo "It is better to use absolute path."
echo "=============================================================================================================="
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
paste $DATASET_PATH/train.de.tok  $DATASET_PATH/train.en.tok > $DATASET_PATH/train.all
python ../src/create_data.py --input_file $DATASET_PATH/train.all --num_splits 8 --src_vocab_file $DATASET_PATH/vocab.de --trg_vocab_file $DATASET_PATH/vocab.en --output_file $OUTPUT_PATH/multi30k_train_mindrecord --max_seq_length 32 --bucket [32]
paste $DATASET_PATH/test.de.tok  $DATASET_PATH/test.en.tok > $DATASET_PATH/test.all
python ../src/create_data.py --input_file $DATASET_PATH/test.all --num_splits 1 --src_vocab_file $DATASET_PATH/vocab.de --trg_vocab_file $DATASET_PATH/vocab.en --output_file $OUTPUT_PATH/multi30k_test_mindrecord --max_seq_length 32 --bucket [32]
