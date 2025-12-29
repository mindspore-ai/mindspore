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

if [ $# -lt 1 ]; then
  echo "Usage: $0 python_file"
  exit 1
fi

LOG_PATH="ascend_log"
export ASCEND_PROCESS_LOG_PATH=`pwd`/${LOG_PATH}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_GLOBAL_EVENT_ENABLE=0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MS_DEV_DUMP_IR_PASSES="silent_check,graph_build"

export HCCL_EXEC_TIMEOUT=80

# NOTE: environment variable `NPU_ASD_ENABLE` is set in python script

rm -rf ${LOG_PATH}
rm -rf ms_graphs
rm -rf worker_*.log

msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --join=True "$@"
