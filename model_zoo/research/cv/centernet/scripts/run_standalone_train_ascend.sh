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
echo "Please run the scipt as: "
echo "bash run_standalone_pretrain_ascend.sh DEVICE_ID EPOCH_SIZE"
echo "for example: bash run_standalone_pretrain_ascend.sh 0 350"
echo "=============================================================================================================="

DEVICE_ID=$1
EPOCH_SIZE=$2

mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python ${PROJECT_DIR}/../train.py  \
    --distribute=false \
    --need_profiler=false \
    --profiler_path=./profiler \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --enable_save_ckpt=true \
    --do_shuffle=true \
    --enable_data_sink=true \
    --data_sink_steps=50 \
    --load_checkpoint_path="" \
    --save_checkpoint_steps=10000 \
    --save_checkpoint_num=1 \
    --mindrecord_dir="" \
    --mindrecord_prefix="coco_hp.train.mind" \
    --visual_image=false \
    --save_result_dir="" > training_log.txt 2>&1 &