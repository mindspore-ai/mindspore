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

if [ $# != 2 ]
then 
    echo "Usage: sh run_train.sh [SRC_PATH] [DATASET_NAME]"
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

DATASET_NAME=$2
echo $DATASET_NAME

if [ ! -d data_mr ]; then
  mkdir data_mr
else
  echo data_mr exist
fi
MINDRECORD_PATH=`pwd`/data_mr

rm -f $MINDRECORD_PATH/$DATASET_NAME
rm -f $MINDRECORD_PATH/$DATASET_NAME.db

cd ../../../example/graph_to_mindrecord || exit

python writer.py --mindrecord_script $DATASET_NAME \
--mindrecord_file "$MINDRECORD_PATH/$DATASET_NAME" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "$SRC_PATH"

cd - || exit
