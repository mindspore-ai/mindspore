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

RUN_MODE=$1
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
BASE_PATH=$(cd "$(dirname $0)"; pwd)
rm -rf ./*.log
python ${BASE_PATH}/test_add.py --run_mode=$RUN_MODE > ms.log 2>&1 &
process_pid=`echo $!`
wait ${process_pid}
status=$(echo $?)
if [ "${status}" != "0" ]; then
    exit 1
fi

need_str='aclrtCtxSetSysParamOpt, opt = 1, value = 1'
result=$(cat ms.log | grep -E "${need_str}")

if [ "$result" == "" ]; then
    exit 1
fi
exit 0

