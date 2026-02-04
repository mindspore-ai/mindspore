#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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
echo "=========================================="
echo "Please run the script as: "
echo "bash msrun_single.sh <DATA_PATH> <DUMP_CONFIG_PATH>"
echo "==========================================="

DUMP_CONFIG_PATH=$1

echo "DUMP CONFIG PATH IS " $DUMP_CONFIG_PATH
rm -rf msrun_log
mkdir msrun_log

echo "start training"

export MINDSPORE_DUMP_CONFIG=$DUMP_CONFIG_PATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
msrun --worker_num=4 --local_worker_num=4 --master_port=8752 --log_dir=msrun_log --join=True --cluster_time_out=300 resnet.py
