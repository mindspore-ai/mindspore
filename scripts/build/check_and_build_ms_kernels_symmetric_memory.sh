#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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

package_check() {
  absolute_path=$1
  relative_path=${absolute_path#${BASEPATH}/}
  expected_sha256=`git show HEAD:${relative_path} | grep sha256 | awk -F: '{print $NF}'`
  actual_sha256=`sha256sum ${absolute_path} | awk '{print $1}'`
  if [[ "${expected_sha256}" != "${actual_sha256}" ]]; then
    echo "[ERROR] SHA256 hash of ${absolute_path} does not match expected value"
    echo "expected: ${expected_sha256}"
    echo "actual: ${actual_sha256}"
    exit 1
  fi
}

if [[ -n "${MS_SYMMETRIC_MEMORY_KERNEL_HOME}" ]]; then
  echo "Use local MS_SYMMETRIC_MEMORY_KERNEL_HOME : ${MS_SYMMETRIC_MEMORY_KERNEL_HOME}"
  return
fi
if [[ "$(uname)" != Linux || ("$(arch)" != x86_64 && "$(arch)" != aarch64) ]]; then
  echo "[WARNING] symmetric memory kernels only supports linux system, x86_64 or aarch64 CPU arch."
  return
fi
file_path=${BASEPATH}/mindspore/ops/kernel/ascend/symmetric_memory/prebuild/$(arch)

symmetric_memory_file_name=${file_path}/ms_kernels_symmetric_memory.tar.gz
if [[ ! -f "${symmetric_memory_file_name}" ]]; then
  echo "[WARNING] The file ${symmetric_memory_file_name}  does NOT EXIST."
  return
fi
symmetric_memory_file_lines=`cat "${symmetric_memory_file_name}" | wc -l`
if [[ ${symmetric_memory_file_lines} -eq 3 ]]; then
  echo "[WARNING] The file ms_kernels_symmetric_memory.tar.gz is not pulled. Please ensure git-lfs is installed by"
  echo "[WARNING] 'git lfs install' and retry downloading using 'git lfs pull'."
  return
fi

tar --warning=no-unknown-keyword -zxf ${symmetric_memory_file_name} -C ${file_path}
if [[ $? -ne 0 ]]; then
  echo "[WARNING] Unzip ms_kernels_symmetric_memory.tar.gz FAILED!"
  return
fi
echo "Unzip ms_kernels_symmetric_memory.tar.gz SUCCESS!"

export MS_SYMMETRIC_MEMORY_KERNEL_HOME="${file_path}/ms_kernels_symmetric_memory"
echo "MS_SYMMETRIC_MEMORY_KERNEL_HOME = ${MS_SYMMETRIC_MEMORY_KERNEL_HOME}"
