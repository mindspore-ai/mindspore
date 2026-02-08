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

#ifndef CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_BUILDER_MANAGER_H_
#define CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_BUILDER_MANAGER_H_
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_builder.h"
#include <map>
#include <utility>
#include <memory>
#include <string>
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace kernel {
using GraphKernelBuildCreator = std::function<std::shared_ptr<GraphKernelBuilder>()>;

class RUNTIME_HARDWARE_EXPORT GraphKernelBuildManager {
 public:
  static GraphKernelBuildManager &Instance();
  void Register(const std::string &device_type, bool is_dynamic, GraphKernelBuildCreator &&creator);
  void Clear() { base_map_.clear(); }
  std::shared_ptr<GraphKernelBuilder> GetGraphKernelBuilder(const std::string &device_type, bool is_dynamic);

 private:
  std::map<std::pair<std::string, bool>, GraphKernelBuildCreator> base_map_;
};

class GraphKernelBuildRegister {
 public:
  GraphKernelBuildRegister(const std::string &device_type, bool is_dynamic, GraphKernelBuildCreator &&creator) {
    GraphKernelBuildManager::Instance().Register(device_type, is_dynamic, std::move(creator));
  }
  ~GraphKernelBuildRegister() = default;
};

#define REG_GRAPH_KERNEL_BUILDER(DEVICE_TYPE, IS_DYNAMIC, BUILDER_CLASS)                           \
  static const GraphKernelBuildRegister g_graph_kernel_builder_##DEVICE_TYPE##_##IS_DYNAMIC##_reg( \
    DEVICE_TYPE, IS_DYNAMIC, []() { return std::make_shared<BUILDER_CLASS>(); });
}  // namespace kernel
}  // namespace mindspore

#endif  // CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_AKG_AKG_KERNEL_BUILD_MANAGER_H_
