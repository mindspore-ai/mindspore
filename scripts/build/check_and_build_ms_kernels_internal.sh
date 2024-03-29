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

if [[ "$(uname)" == Linux && "$(arch)" == aarch64 ]]; then
  if [ -n "${MS_INTERNAL_KERNEL_HOME}" ]; then
    echo "Use local MS_INTERNAL_KERNEL_HOME : ${MS_INTERNAL_KERNEL_HOME}"
  else
    file_path=${BASEPATH}/mindspore/ccsrc/plugin/device/ascend/kernel/internal/prebuild
    lib_file=${file_path}/ms_kernels_internal.tar.gz
    if [ -f "${lib_file}" ]; then
      file_lines=`cat "${lib_file}" | wc -l`
      if [ ${file_lines} -ne 3 ]; then
        tar -zxf ${lib_file} -C ${file_path}
        if [ $? -eq 0 ]; then
          echo "Unzip ms_kernel_internal.tar.gz SUCCESS!"
          export MS_INTERNAL_KERNEL_HOME="${file_path}/ms_kernels_internal"
          echo "MS_INTERNAL_KERNEL_HOME = ${MS_INTERNAL_KERNEL_HOME}"
        else
          echo "[WARNING] Unzip ms_kernel_internal.tar.gz FAILED!"
        fi
      else
        echo "[WARNING] The file ms_kernel_internal.tar.gz is not pulled. Please ensure git-lfs is installed by"
        echo "[WARNING] 'git lfs install' and retry downloading using 'git lfs pull'."
      fi
    else
      echo "[WARNING] The file ms_kernel_internal.tar.gz does NOT EXIST."
    fi
  fi
fi
