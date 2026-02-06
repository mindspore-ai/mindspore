/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_comm_info_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace graphkernel {
GraphKernelCommInfoManager &GraphKernelCommInfoManager::Instance() {
  static GraphKernelCommInfoManager instance{};
  return instance;
}

void GraphKernelCommInfoManager::Register(const std::string &device_type, GraphKernelCommInfoCreator &&creator) {
  if (comm_info_map_.find(device_type) == comm_info_map_.end()) {
    (void)comm_info_map_.emplace(device_type, creator);
  }
}

std::shared_ptr<GraphKernelCommInfo> GraphKernelCommInfoManager::GetCommInfo(const std::string &device_type) {
  auto iter = comm_info_map_.find(device_type);
  if (comm_info_map_.end() != iter) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  return nullptr;
}
}  // namespace graphkernel
}  // namespace mindspore
