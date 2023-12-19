/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/host/host_kernel_mod.h"

#include "utils/ms_context.h"
#include "kernel/common_utils.h"
#include "kernel/framework_utils.h"

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

bool HostKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  return true;
}

int HostKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  input_size_list_.clear();
  output_size_list_.clear();

  auto calc_size_list = [](const std::vector<KernelTensor *> &tensors, std::vector<size_t> *list_ptr) -> bool {
    for (KernelTensor *tensor : tensors) {
      int64_t size = 1;
      if (!GetShapeSize(tensor->GetShapeVector(), tensor->dtype(), &size)) {
        return false;
      }
      list_ptr->push_back(LongToSize(size));
    }
    return true;
  };

  if (!calc_size_list(inputs, &input_size_list_)) {
    return KRET_RESIZE_FAILED;
  }
  if (!calc_size_list(outputs, &output_size_list_)) {
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}
}  // namespace kernel
}  // namespace mindspore
