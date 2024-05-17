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

if [[ -n "${MS_INTERNAL_KERNEL_HOME}" ]]; then
  echo "Use local MS_INTERNAL_KERNEL_HOME : ${MS_INTERNAL_KERNEL_HOME}"
  return
fi
if [[ "$(uname)" != Linux || ("$(arch)" != x86_64 && "$(arch)" != aarch64) ]]; then
  echo "[WARNING] Internal kernels only supports linux system, x86_64 or aarch64 CPU arch."
  return
fi
file_path=${BASEPATH}/mindspore/ccsrc/plugin/device/ascend/kernel/internal/prebuild/$(arch)
file_name=${file_path}/ms_kernels_internal.tar.gz
if [[ ! -f "${file_name}" ]]; then
  echo "[WARNING] The file ${file_name}  does NOT EXIST."
  return
fi
file_lines=`cat "${file_name}" | wc -l`
if [[ ${file_lines} -eq 3 ]]; then
  echo "[WARNING] The file ms_kernel_internal.tar.gz is not pulled. Please ensure git-lfs is installed by"
  echo "[WARNING] 'git lfs install' and retry downloading using 'git lfs pull'."
  return
fi
tar -zxf ${file_name} -C ${file_path}
if [[ $? -ne 0 ]]; then
  echo "[WARNING] Unzip ms_kernel_internal.tar.gz FAILED!"
  return
fi
echo "Unzip ms_kernel_internal.tar.gz SUCCESS!"
export MS_INTERNAL_KERNEL_HOME="${file_path}/ms_kernels_internal"
echo "MS_INTERNAL_KERNEL_HOME = ${MS_INTERNAL_KERNEL_HOME}"