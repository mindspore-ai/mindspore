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

arch_name=`uname -m`
lib_file="${BASEPATH}/mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/${arch_name}/libdvm.a"
if [ -f "${lib_file}" ]; then
  file_lines=`cat "${lib_file}" | wc -l`
  if [ ${file_lines} -ne 3 ]; then
    export ENABLE_DVM="on"
    export DVM_LIB="${lib_file}"
  fi
fi
