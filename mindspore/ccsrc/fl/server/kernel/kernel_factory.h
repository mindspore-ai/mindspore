/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_KERNEL_FACTORY_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include "fl/server/common.h"
#include "fl/server/kernel/params_info.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// KernelFactory is used to select and build kernels in server. It's the base class of OptimizerKernelFactory
// and AggregationKernelFactory.

// Unlike normal MindSpore operator kernels, the server defines multiple types of kernels. For example: Aggregation
// Kernel, Optimizer Kernel, Forward Kernel, etc. So we define KernelFactory as a template class for register of all
// types of kernels.

// Because most information we need to create a server kernel is in func_graph passed by the front end, we create a
// server kernel based on a cnode.

// Typename K refers to the shared_ptr of the kernel type.
// Typename C refers to the creator function of the kernel.
template <typename K, typename C>
class KernelFactory {
 public:
  KernelFactory() = default;
  virtual ~KernelFactory() = default;

  static KernelFactory &GetInstance() {
    static KernelFactory instance;
    return instance;
  }

  // Kernels are registered by parameter information and its creator(constructor).
  void Register(const std::string &name, const ParamsInfo &params_info, C &&creator) {
    name_to_creator_map_[name].push_back(std::make_pair(params_info, creator));
  }

  // The kernels in server are created from func_graph's kernel_node passed by the front end.
  K Create(const std::string &name, const CNodePtr &kernel_node) {
    if (name_to_creator_map_.count(name) == 0) {
      MS_LOG(ERROR) << "Creating kernel failed: " << name << " is not registered.";
    }
    for (const auto &name_type_creator : name_to_creator_map_[name]) {
      const ParamsInfo &params_info = name_type_creator.first;
      const C &creator = name_type_creator.second;
      if (Matched(params_info, kernel_node)) {
        auto kernel = creator();
        kernel->set_params_info(params_info);
        return kernel;
      }
    }
    return nullptr;
  }

 private:
  KernelFactory(const KernelFactory &) = delete;
  KernelFactory &operator=(const KernelFactory &) = delete;

  // Judge whether the server kernel can be created according to registered ParamsInfo.
  virtual bool Matched(const ParamsInfo &params_info, const CNodePtr &kernel_node) { return true; }

  // Generally, a server kernel can correspond to several ParamsInfo which is registered by the method 'Register' in
  // server kernel's *.cc files.
  std::unordered_map<std::string, std::vector<std::pair<ParamsInfo, C>>> name_to_creator_map_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_KERNEL_FACTORY_H_
