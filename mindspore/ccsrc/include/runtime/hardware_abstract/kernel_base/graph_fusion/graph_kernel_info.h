/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_INFO_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_INFO_H_
#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
class GraphKernelInfo {
 public:
  GraphKernelInfo() = default;
  virtual ~GraphKernelInfo() = default;
  virtual void SetKernelInfo(const CNodePtr &, KernelType) {}
};

using GraphKernelInfoCreator = std::function<std::shared_ptr<GraphKernelInfo>()>;

class RUNTIME_HARDWARE_EXPORT GraphKernelInfoManager {
 public:
  static GraphKernelInfoManager &Instance();
  void Register(const std::string &device_type, GraphKernelInfoCreator &&creator);
  void Clear();
  std::shared_ptr<GraphKernelInfo> GetGraphKernelInfo(const std::string &device_type);

 private:
  std::map<std::string, GraphKernelInfoCreator> base_map_;
};

class RUNTIME_HARDWARE_EXPORT GraphKernelInfoRegister {
 public:
  GraphKernelInfoRegister(const std::string &device_type, GraphKernelInfoCreator &&creator);
  ~GraphKernelInfoRegister() = default;
};

#define REG_GRAPH_KERNEL_INFO(DEVICE_TYPE, KERNEL_CLASS)                           \
  static const GraphKernelInfoRegister g_graph_kernel_info_##DEVICE_TYPE##_##_reg( \
    DEVICE_TYPE, []() { return std::make_shared<KERNEL_CLASS>(); })
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_INFO_H_
