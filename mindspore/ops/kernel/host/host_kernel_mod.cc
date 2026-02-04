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

#include "kernel/host/host_kernel_mod.h"

#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
void HostKernelFactory::Register(const std::string &name, HostKernelCreater &&fun) {
  hostKernelMap_.emplace(name, std::move(fun));
}

std::shared_ptr<HostKernelMod> HostKernelFactory::Get(const std::string &name) {
  const auto &map = Get().hostKernelMap_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

HostKernelFactory &HostKernelFactory::Get() {
  static HostKernelFactory instance{};
  return instance;
}

int HostKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (UseSimulationApi()) {
    return ret;
  }
  GetWorkSpaceInfo(inputs, outputs);
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->tensor_storage_info() != nullptr) {
      output_size_list_[i] =
        SizeOf(outputs[i]->tensor_storage_info()->ori_shape) * abstract::TypeIdSize(outputs[i]->dtype_id());
    }
  }
  return ret;
}
}  // namespace kernel
}  // namespace mindspore
