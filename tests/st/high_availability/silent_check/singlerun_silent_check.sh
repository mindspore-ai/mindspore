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

if [ $# -lt 1 ]; then
  echo "Usage: $0 python_file"
  exit 1
fi

py_file=$1
shift

if [ "x${ASCEND_PROCESS_LOG_PATH}" == "x" ]; then
  LOG_PATH="ascend_log"
  export ASCEND_PROCESS_LOG_PATH=`pwd`/${LOG_PATH}
fi
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_GLOBAL_EVENT_ENABLE=0

# NOTE: set environment variable `NPU_ASD_ENABLE` to `3` to enable silent_check
if [ "x${NPU_ASD_ENABLE}" == "x" ]; then
  export NPU_ASD_ENABLE=3
fi

rm -rf ${LOG_PATH}

python -u ${py_file} "$@"
