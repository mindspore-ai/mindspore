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
echo "bash run.sh DATA_PATH RANK_TABLE"
echo "For example: bash run.sh /path/dataset /path/rank_table"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
RANK_SIZE=8
RANK_TABLE=$(get_real_path $1)

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export RANK_TABLE_FILE=$RANK_TABLE

start_divice=0
for((i=$start_divice;i<$[$start_divice+$RANK_SIZE];i++))
do
    rm -rf device$i
    mkdir device$i
    mkdir device$i/src
    cp ./train.py  ./device$i
    cp ./src/net.py ./src/loss.py ./src/config.py ./src/util.py ./src/data_loader.py ./src/generate_anchors.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python3 train.py  --is_parallel=True &> log &
    cd ../
done
echo "finish"
cd ../
