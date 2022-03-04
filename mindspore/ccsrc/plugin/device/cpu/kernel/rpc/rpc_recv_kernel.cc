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

#include "plugin/device/cpu/kernel/rpc/rpc_recv_kernel.h"
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, RpcRecvKernelMod::RpcRecvFunc>> RpcRecvKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddAllSameAttr(true)
     .AddOutInRef(0, 0),
   &RpcRecvKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> RpcRecvKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, RpcRecvFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RpcRecv, RpcRecvKernelMod);
}  // namespace kernel
}  // namespace mindspore
