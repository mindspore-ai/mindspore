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
echo "bash run_train_cpu.sh ACLIMDB_DIR GLOVE_DIR"
echo "for example: bash run_train_gpu.sh ./aclimdb ./glove_dir"
echo "=============================================================================================================="

ACLIMDB_DIR=$1
GLOVE_DIR=$2

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python train.py  \
    --device_target="CPU" \
    --aclimdb_path=$ACLIMDB_DIR \
    --glove_path=$GLOVE_DIR \
    --preprocess=true  \
    --preprocess_path=./preprocess > log.txt 2>&1 &
