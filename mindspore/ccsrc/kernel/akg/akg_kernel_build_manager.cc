/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/akg/akg_kernel_build_manager.h"
#include <memory>
namespace mindspore {
namespace kernel {
AkgKernelBuildManager &AkgKernelBuildManager::Instance() {
  static AkgKernelBuildManager instance{};
  return instance;
}

void AkgKernelBuildManager::Register(const std::string &device_type, AkgKernelBuildCreator &&creator) {
  if (base_map_.find(device_type) == base_map_.end()) {
    (void)base_map_.emplace(device_type, creator);
  }
}

std::shared_ptr<AkgKernelBuilder> AkgKernelBuildManager::GetAkgKernelBuilder(const std::string &device_type) {
  auto iter = base_map_.find(device_type);
  if (base_map_.end() != iter) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  return nullptr;
}
}  // namespace kernel
}  // namespace mindspore
