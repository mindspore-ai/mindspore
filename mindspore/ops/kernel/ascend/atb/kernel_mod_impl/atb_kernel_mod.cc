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

#include "kernel/ascend/atb/kernel_mod_impl/atb_kernel_mod.h"
#include <algorithm>
#include <vector>
#include <functional>
#include "utils/ms_context.h"

namespace mindspore::kernel {
ATBKernelMod::~ATBKernelMod() {
  if (!UseSimulationApi()) {
    atb::DestroyOperation(op_);
  }
}

bool ATBKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "ATBKernelMod Init";
  return true;
}

int ATBKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (UseSimulationApi()) {
    return ret;
  }
  GetWorkSpaceInfo(inputs, outputs);
  return ret;
}

void ATBKernelMod::UpdateWorkspace(uint64_t workspace_size) {
  workspace_size_list_.clear();
  workspace_size_list_ = {workspace_size};
}
}  // namespace mindspore::kernel
