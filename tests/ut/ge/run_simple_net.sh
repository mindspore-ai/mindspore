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
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
rm -rf ${BASE_PATH}/simple_net
mkdir ${BASE_PATH}/simple_net
echo "start test simple net with ge backend"
cd ${BASE_PATH}/simple_net
env > env.log
python ../run_simple_net.py > test_simple_net.log 2>&1 &
process_pid=`echo $!`
wait ${process_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test simple net with ge backend failed. status: ${status}"
    exit 1
else
    echo "[INFO] test simple net with ge backend success."
fi

exit 0
