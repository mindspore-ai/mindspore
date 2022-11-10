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

#ifndef MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_INFO_H_
#define MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_INFO_H_
#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include "ir/dtype.h"
#include "ir/kernel_info_dev.h"
#include "kernel/kernel.h"
#include "include/backend/visible.h"
namespace mindspore {
class GraphKernelInfo {
 public:
  GraphKernelInfo() = default;
  virtual ~GraphKernelInfo() = default;
  virtual void SetKernelInfo(const CNodePtr &, KernelType) {}
};

using GraphKernelInfoCreator = std::function<std::shared_ptr<GraphKernelInfo>()>;

class BACKEND_EXPORT GraphKernelInfoManager {
 public:
  static GraphKernelInfoManager &Instance() {
    static GraphKernelInfoManager instance{};
    return instance;
  }
  void Register(const std::string &device_type, GraphKernelInfoCreator &&creator) {
    if (base_map_.find(device_type) == base_map_.end()) {
      (void)base_map_.emplace(device_type, creator);
    }
  }
  void Clear() { base_map_.clear(); }
  std::shared_ptr<GraphKernelInfo> GetGraphKernelInfo(const std::string &device_type) {
    auto iter = base_map_.find(device_type);
    if (base_map_.end() != iter) {
      MS_EXCEPTION_IF_NULL(iter->second);
      return (iter->second)();
    }
    MS_LOG(WARNING) << "Can not get a graph kernel info ptr on device: " << device_type;
    return nullptr;
  }

 private:
  std::map<std::string, GraphKernelInfoCreator> base_map_;
};

class GraphKernelInfoRegister {
 public:
  GraphKernelInfoRegister(const std::string &device_type, GraphKernelInfoCreator &&creator) {
    GraphKernelInfoManager::Instance().Register(device_type, std::move(creator));
  }
  ~GraphKernelInfoRegister() = default;
};

#define REG_GRAPH_KERNEL_INFO(DEVICE_TYPE, KERNEL_CLASS)                           \
  static const GraphKernelInfoRegister g_graph_kernel_info_##DEVICE_TYPE##_##_reg( \
    DEVICE_TYPE, []() { return std::make_shared<KERNEL_CLASS>(); });
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_INFO_H_
