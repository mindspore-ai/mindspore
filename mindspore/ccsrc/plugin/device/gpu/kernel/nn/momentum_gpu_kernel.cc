/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/momentum_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S, typename G>
void MomentumGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, void *stream_ptr) {
  T *variable = GetDeviceAddress<T>(inputs, kIndex0);
  T *accumulation = GetDeviceAddress<T>(inputs, kIndex1);
  S *learning_rate = GetDeviceAddress<S>(inputs, kIndex2);
  G *gradient = GetDeviceAddress<G>(inputs, kIndex3);
  S *momentum = GetDeviceAddress<S>(inputs, kIndex4);
  MomentumUpdateVariable(inputs[kIndex0]->size / sizeof(T), variable, accumulation, learning_rate, gradient, momentum,
                         use_nesterov_, reinterpret_cast<cudaStream_t>(stream_ptr));
}

std::vector<std::pair<KernelAttr, MomentumGpuKernelMod::LaunchFunc>> MomentumGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutInRef(0, 0),
   &MomentumGpuKernelMod::LaunchKernel<float, float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutInRef(0, 0),
   &MomentumGpuKernelMod::LaunchKernel<half, half, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutInRef(0, 0),
   &MomentumGpuKernelMod::LaunchKernel<half, float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutInRef(0, 0),
   &MomentumGpuKernelMod::LaunchKernel<float, float, half>}};

std::vector<KernelAttr> MomentumGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyMomentum, MomentumGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
