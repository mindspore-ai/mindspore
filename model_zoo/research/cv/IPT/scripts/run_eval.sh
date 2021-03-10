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

export DEVICE_ID=$1
DATA_DIR=$2
DATA_SET=$3
PATH_CHECKPOINT=$4

python ../eval.py --dir_data=$DATA_DIR --data_test=$DATA_SET --nochange --test_only --ext img --chop_new --scale 4 --pth_path=$PATH_CHECKPOINT > eval.log 2>&1 &
python ../eval.py --dir_data=$DATA_DIR --data_test=$DATA_SET --nochange --test_only --ext img --chop_new --scale 3 --pth_path=$PATH_CHECKPOINT > eval.log 2>&1 &
python ../eval.py --dir_data=$DATA_DIR --data_test=$DATA_SET --nochange --test_only --ext img --chop_new --scale 2 --pth_path=$PATH_CHECKPOINT > eval.log 2>&1 &

##denoise
python ../eval.py --dir_data=$DATA_DIR --data_test=$DATA_SET --nochange --test_only --ext img --chop_new --scale 1 --denoise --sigma 30 --pth_path=$PATH_CHECKPOINT > eval.log 2>&1 &
python ../eval.py --dir_data=$DATA_DIR --data_test=$DATA_SET --nochange --test_only --ext img --chop_new --scale 1 --denoise --sigma 50 --pth_path=$PATH_CHECKPOINT > eval.log 2>&1 &

##derain
python ../eval.py --dir_data=$DATA_DIR --data_test=$DATA_SET --nochange --test_only --ext img --chop_new --scale 1 --derain  --derain_test 1 --pth_path=$PATH_CHECKPOINT > eval.log 2>&1 &