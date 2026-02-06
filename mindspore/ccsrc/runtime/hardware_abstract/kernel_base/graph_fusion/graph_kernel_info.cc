/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_info.h"

namespace mindspore {
GraphKernelInfoManager &GraphKernelInfoManager::Instance() {
  static GraphKernelInfoManager instance{};
  return instance;
}

void GraphKernelInfoManager::Register(const std::string &device_type, GraphKernelInfoCreator &&creator) {
  if (base_map_.find(device_type) == base_map_.end()) {
    (void)base_map_.emplace(device_type, creator);
  }
}
void GraphKernelInfoManager::Clear() { base_map_.clear(); }
std::shared_ptr<GraphKernelInfo> GraphKernelInfoManager::GetGraphKernelInfo(const std::string &device_type) {
  auto iter = base_map_.find(device_type);
  if (base_map_.end() != iter) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  MS_LOG(WARNING) << "Can not get a graph kernel info ptr on device: " << device_type;
  return nullptr;
}

GraphKernelInfoRegister::GraphKernelInfoRegister(const std::string &device_type, GraphKernelInfoCreator &&creator) {
  GraphKernelInfoManager::Instance().Register(device_type, std::move(creator));
}
}  // namespace mindspore
