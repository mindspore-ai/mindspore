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

#include "plugin/device/cpu/kernel/rpc/rpc_send_kernel.h"

namespace mindspore {
namespace kernel {
void RpcSendKernelMod::Init(const CNodePtr &kernel_node) {
  DeprecatedNativeCpuKernelMod::Init(kernel_node);
  // Assign workspace memory with the same size of inputs. It's the data which will be sent to remote.
  size_t total_size = 0;
  total_size = std::accumulate(input_size_list_.begin(), input_size_list_.end(), total_size,
                               [](size_t total_size, const auto &input_size) { return total_size + input_size; });
  workspace_size_list_.push_back(total_size);
}

std::vector<KernelAttr> RpcSendKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true).AddAllOutInRef(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RpcSend, RpcSendKernelMod);
}  // namespace kernel
}  // namespace mindspore
