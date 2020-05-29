#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_eval.sh DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash run_eval.sh 0 /path/zh-wiki/ "
echo "=============================================================================================================="
 
DEVICE_ID=$1
DATA_DIR=$2
 
mkdir -p ms_log 
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python evaluation.py  \
    --device_id=$DEVICE_ID \
    --checkpoint_url="" \
    --data_url=$DATA_DIR > log.txt 2>&1 &