#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

if [ $# != 1 ]
then
    echo "Usage: sh run_process_data_ascend.sh [SRC_PATH] "
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
SRC_PATH=$(get_real_path $1)
echo $SRC_PATH


if [ ! -d data_mr ]; then
  mkdir data_mr
else
  echo data_mr exist
fi
MINDRECORD_PATH=`pwd`/data_mr

rm -rf ${MINDRECORD_PATH:?}/*
INTER_FILE_DIR=$MINDRECORD_PATH/InterFile
mkdir -p $INTER_FILE_DIR

cd ../../../../utils/graph_to_mindrecord || exit

echo "Start to converting data."
python amazon_beauty/converting_data.py --src_path $SRC_PATH --out_path $INTER_FILE_DIR

echo "Start to generate train_mr."
python writer.py --mindrecord_script amazon_beauty \
--mindrecord_file "$MINDRECORD_PATH/train_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "$INTER_FILE_DIR/user.csv:$INTER_FILE_DIR/item.csv:$INTER_FILE_DIR/rating_train.csv"

echo "Start to generate test_mr."
python writer.py --mindrecord_script amazon_beauty \
--mindrecord_file "$MINDRECORD_PATH/test_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "$INTER_FILE_DIR/user.csv:$INTER_FILE_DIR/item.csv:$INTER_FILE_DIR/rating_test.csv"

for id in {0..4}
do
    echo "Start to generate sampled${id}_mr."
    python writer.py --mindrecord_script amazon_beauty \
    --mindrecord_file "${MINDRECORD_PATH}/sampled${id}_mr" \
    --mindrecord_partitions 1 \
    --mindrecord_header_size_by_bit 18 \
    --mindrecord_page_size_by_bit 20 \
    --graph_api_args "$INTER_FILE_DIR/user.csv:$INTER_FILE_DIR/item.csv:$INTER_FILE_DIR/rating_sampled${id}.csv"
done

