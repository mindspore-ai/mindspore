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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_FACTORY_H_

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include "fl/server/common.h"
#include "fl/server/kernel/round/round_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using RoundKernelCreator = std::function<std::shared_ptr<RoundKernel>()>;
// Kernel factory of round kernels.
class RoundKernelFactory {
 public:
  static RoundKernelFactory &GetInstance();
  void Register(const std::string &name, RoundKernelCreator &&creator);
  std::shared_ptr<RoundKernel> Create(const std::string &name);

 private:
  RoundKernelFactory() = default;
  ~RoundKernelFactory() = default;
  RoundKernelFactory(const RoundKernelFactory &) = delete;
  RoundKernelFactory &operator=(const RoundKernelFactory &) = delete;

  std::unordered_map<std::string, RoundKernelCreator> name_to_creator_map_;
};

class RoundKernelRegister {
 public:
  RoundKernelRegister(const std::string &name, RoundKernelCreator &&creator) {
    RoundKernelFactory::GetInstance().Register(name, std::move(creator));
  }
  ~RoundKernelRegister() = default;
};

#define REG_ROUND_KERNEL(NAME, CLASS)                                                        \
  static_assert(std::is_base_of<RoundKernel, CLASS>::value, " must be base of RoundKernel"); \
  static const RoundKernelRegister g_##NAME##_round_kernel_reg(#NAME, []() { return std::make_shared<CLASS>(); });
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_FACTORY_H_
