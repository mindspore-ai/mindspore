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

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
DEVICE_TARGET=$1

python ${self_path}/test_dynamic_embedding_standalone.py                          \
       --device_target=$DEVICE_TARGET &> ${self_path}/dynamic_embedding.log 2>&1 &
pid=`echo $!`

wait ${pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test dynamic embedding standalone failed, status: ${status}"
    exit 1
fi
