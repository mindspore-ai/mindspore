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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_train_gpu.sh DEVICE_ID ACLIMDB_DIR GLOVE_DIR CKPT_FILE"
echo "for example: bash run_train_gpu.sh 0 ./aclimdb ./glove_dir lstm-20_390.ckpt"
echo "=============================================================================================================="

DEVICE_ID=$1
ACLIMDB_DIR=$2
GLOVE_DIR=$3
CKPT_FILE=$4

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python eval.py  \
    --device_target="GPU" \
    --aclimdb_path=$ACLIMDB_DIR \
    --glove_path=$GLOVE_DIR \
    --preprocess=false  \
    --preprocess_path=./preprocess \
    --ckpt_path=$CKPT_FILE > log.txt 2>&1 &
