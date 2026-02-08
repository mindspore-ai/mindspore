/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_builder_manager.h"
#include <memory>
namespace mindspore {
namespace kernel {
GraphKernelBuildManager &GraphKernelBuildManager::Instance() {
  static GraphKernelBuildManager instance{};
  return instance;
}

void GraphKernelBuildManager::Register(const std::string &device_type, bool is_dynamic,
                                       GraphKernelBuildCreator &&creator) {
  auto idx = std::make_pair(device_type, is_dynamic);
  if (base_map_.find(idx) == base_map_.end()) {
    (void)base_map_.emplace(idx, creator);
  }
}

std::shared_ptr<GraphKernelBuilder> GraphKernelBuildManager::GetGraphKernelBuilder(const std::string &device_type,
                                                                                   bool is_dynamic) {
  auto idx = std::make_pair(device_type, is_dynamic);
  auto iter = base_map_.find(idx);
  if (base_map_.end() != iter) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  return nullptr;
}
}  // namespace kernel
}  // namespace mindspore
