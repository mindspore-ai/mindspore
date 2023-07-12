/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_manager.h"

namespace mindspore {
namespace kernel {
void *AkgCpuKernelManagerAbs::SearchFunc(const std::string &kernel_name) const {
  auto iter = cpu_func_map_.find(kernel_name);
  if (iter == cpu_func_map_.end()) {
    return nullptr;
  } else {
    return iter->second.first;
  }
}

void *AkgCpuKernelManagerAbs::SearchFuncWithSharedLock(const std::string &kernel_name) const {
  std::shared_lock lock(mutex_);
  return SearchFunc(kernel_name);
}
}  // namespace kernel
}  // namespace mindspore
