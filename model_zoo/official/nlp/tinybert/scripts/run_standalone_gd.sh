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
echo "bash scripts/run_standalone_gd.sh"
echo "for example: bash scripts/run_standalone_gd.sh"
echo "running....... please see details by log.txt"
echo "=============================================================================================================="


mkdir -p ms_log
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_general_distill.py  \
    --distribute="false" \
    --device_target="Ascend" \
    --epoch_size=3 \
    --device_id=0 \
    --enable_data_sink="true" \
    --data_sink_steps=100 \
    --save_ckpt_step=100 \
    --max_ckpt_num=1 \
    --save_ckpt_path="" \
    --load_teacher_ckpt_path="" \
    --data_dir="" \
    --schema_dir="" \
    --dataset_type="tfrecord" > log.txt 2>&1 &
