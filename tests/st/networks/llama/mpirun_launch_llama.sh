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
USE_DEVICE_NUM=$1
TEST_MODE=$2
MF_PATH=${BASE_PATH}/../mindformers
sed -i 's/numpy==1.26.4/numpy==1.20.0/' ${MF_PATH}/requirements.txt
pip install -r ${MF_PATH}/requirements.txt


export MS_FORMAT_MODE=1
export MS_GE_TRAIN=1
export MS_ENABLE_REF_MODE=1
export MS_ENABLE_GE=1
export MS_DEV_CELL_REUSE=1
export MS_GE_ATOMIC_CLEAN_POLICY=1

export MS_MEMORY_POOL_RECYCLE=1

mpirun --allow-run-as-root -n ${USE_DEVICE_NUM} \
  python ${BASE_PATH}/parallel_llama.py --${TEST_MODE} True >${TEST_MODE}.log 2>&1
