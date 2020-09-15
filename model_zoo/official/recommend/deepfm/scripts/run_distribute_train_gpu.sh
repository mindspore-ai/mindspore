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
echo "Please run the script as: "
echo "sh scripts/run_distribute_train.sh DEVICE_NUM DATASET_PATH"
echo "for example: sh scripts/run_distribute_train.sh 8 /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in log/output.log"


export RANK_SIZE=$1
DATA_URL=$2

rm -rf log
mkdir ./log
cp *.py ./log
cp -r src ./log
cd ./log || exit
env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python -u train.py \
    --dataset_path=$DATA_URL \
    --ckpt_path="./" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='GPU' \
    --do_eval=True > output.log 2>&1 &
