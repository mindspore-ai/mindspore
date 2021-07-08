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

#include "fl/server/kernel/round/round_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
RoundKernelFactory &RoundKernelFactory::GetInstance() {
  static RoundKernelFactory instance;
  return instance;
}

void RoundKernelFactory::Register(const std::string &name, RoundKernelCreator &&creator) {
  name_to_creator_map_[name] = creator;
}

std::shared_ptr<RoundKernel> RoundKernelFactory::Create(const std::string &name) {
  if (name_to_creator_map_.count(name) == 0) {
    MS_LOG(ERROR) << "Round kernel " << name << " is not registered.";
    return nullptr;
  }
  auto kernel = name_to_creator_map_[name]();
  kernel->set_name(name);
  return kernel;
}
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
