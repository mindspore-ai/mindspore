#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
rm -rf ${BASE_PATH}/if_by_if
mkdir ${BASE_PATH}/if_by_if
unset SLOG_PRINT_TO_STDOUT
cd ${BASE_PATH}/if_by_if
echo "start test if_by_if with ge backend"
env > env.log
python ../run_if_by_if.py > test_if_by_if.log 2>&1 &
process_pid=`echo $!`
wait ${process_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test if_by_if with ge backend failed. status: ${status}"
    exit 1
else
    echo "[INFO] test if_by_if with ge backend success."
fi

exit 0
