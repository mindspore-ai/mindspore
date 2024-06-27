#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
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
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
CONFIG_FILE=$1
USE_DEVICE_NUM=$2
TEST_MODE=$3
MF_PATH=${BASE_PATH}/../mindformers
pip install -r ${MF_PATH}/requirements.txt

export MS_MEMORY_POOL_RECYCLE=1

mpirun --allow-run-as-root -n ${USE_DEVICE_NUM} \
  python ${BASE_PATH}/infer_baichuan.py \
  --yaml_file ${CONFIG_FILE} \
  --test_mode ${TEST_MODE} >${BASE_PATH}/${TEST_MODE}.log 2>&1
