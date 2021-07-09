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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_OPTIMIZER_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_OPTIMIZER_KERNEL_FACTORY_H_

#include <memory>
#include <string>
#include <utility>
#include "fl/server/kernel/kernel_factory.h"
#include "fl/server/kernel/optimizer_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using OptimizerKernelCreator = std::function<std::shared_ptr<OptimizerKernel>()>;
class OptimizerKernelFactory : public KernelFactory<std::shared_ptr<OptimizerKernel>, OptimizerKernelCreator> {
 public:
  static OptimizerKernelFactory &GetInstance() {
    static OptimizerKernelFactory instance;
    return instance;
  }

 private:
  OptimizerKernelFactory() = default;
  ~OptimizerKernelFactory() override = default;
  OptimizerKernelFactory(const OptimizerKernelFactory &) = delete;
  OptimizerKernelFactory &operator=(const OptimizerKernelFactory &) = delete;

  // Judge whether the server optimizer kernel can be created according to registered ParamsInfo.
  bool Matched(const ParamsInfo &params_info, const CNodePtr &kernel_node) override;
};

class OptimizerKernelRegister {
 public:
  OptimizerKernelRegister(const std::string &name, const ParamsInfo &params_info, OptimizerKernelCreator &&creator) {
    OptimizerKernelFactory::GetInstance().Register(name, params_info, std::move(creator));
  }
  ~OptimizerKernelRegister() = default;
};

// Register optimizer kernel with one template type T.
#define REG_OPTIMIZER_KERNEL(NAME, PARAMS_INFO, CLASS, T)                                               \
  static_assert(std::is_base_of<OptimizerKernel, CLASS<T>>::value, " must be base of OptimizerKernel"); \
  static const OptimizerKernelRegister g_##NAME##_##T##_optimizer_kernel_reg(                           \
    #NAME, PARAMS_INFO, []() { return std::make_shared<CLASS<T>>(); });
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_OPTIMIZER_KERNEL_FACTORY_H_
