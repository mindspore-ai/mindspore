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

export DATA_PATH=$1
export CKPT_PATH=$2
export DEVICE_ID=$3
export SLOG_PRINT_TO_STDOUT=1
python ../eval.py --data_dir=$DATA_PATH --checkpoint_path=$CKPT_PATH --device_id=$DEVICE_ID > log_eval 2>&1 &
