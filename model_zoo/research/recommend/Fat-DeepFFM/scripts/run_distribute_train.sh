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
# ===========================================================================

echo "Please run the script as: "
echo "for example: bash scripts/run_distribute_train.sh  /dataset_path 8  scripts/hccl_8p.json False"
echo "After running the script, the network runs in the background, The log will be generated in logx/output.log"

export LANG="zh_CN.UTF-8"
export RANK_SIZE=$2
export RANK_TABLE_FILE=$3
for ((i = 0; i < RANK_SIZE; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  echo "start training for rank $i, device $DEVICE_ID"
  rm -rf Task$i
  mkdir ./Task$i
  cp *.py ./Task$i
  cp -r ./src ./Task$i
  cd ./Task$i || exit
  python train.py --dataset_path=$1 --do_eval=$4 >output.log 2>&1 &
  cd ../
done
