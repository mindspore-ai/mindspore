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

if [ $# != 5 ]
then
    echo "============================================================================================================"
    echo "Please run the script as: "
    echo "bash run_standalone_train.sh [TASK_NAME] [DEVICE_TARGET] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [LOAD_CKPT_PATH]"
    echo "for example: bash run_standalone_train.sh STS-B Ascend /path/sts-b/train.tf_record /path/sts-b/eval.tf_record /path/xxx.ckpt"
    echo "============================================================================================================"
exit 1
fi

echo "===============================================start training==============================================="

task_name=$1
device_target=$2
train_data_dir=$3
eval_data_dir=$4
load_ckpt_path=$5

mkdir -p ms_log
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python ${PROJECT_DIR}/../train.py \
    --task_name=$task_name \
    --device_target=$device_target \
    --device_id=0 \
    --do_eval=True \
    --load_ckpt_path=$load_ckpt_path \
    --train_data_dir=$train_data_dir \
    --eval_data_dir=$eval_data_dir \
    --epoch_num=3 \
    --logging_step=100 \
    --data_sink_steps=100 \
    --save_ckpt_step=100 \
    --do_quant=True > train_log.txt 2>&1 &
