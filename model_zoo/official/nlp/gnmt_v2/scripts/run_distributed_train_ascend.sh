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
current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=/home/workspace/rank_table_8p.json
export MINDSPORE_HCCL_CONFIG_PATH=/home/workspace/rank_table_8p.json

echo $RANK_TABLE_FILE
export RANK_SIZE=8

for((i=0;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py .
    cp -r ../../src .
    cp -r ../../config .
    export RANK_ID=$i
    export DEVICE_ID=$i
	python ../../train.py --config /home/workspace/gnmt_v2/config/config.json > log_gnmt_network${i}.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
