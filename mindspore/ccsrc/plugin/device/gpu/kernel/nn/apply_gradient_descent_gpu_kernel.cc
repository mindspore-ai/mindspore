/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/apply_gradient_descent_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ApplyGradientDescentKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr) {
  T *var = GetDeviceAddress<T>(inputs, kIndex0);
  T *alpha = GetDeviceAddress<T>(inputs, kIndex1);
  T *delta = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  CalApplyGradientDescent(input_size_, var, alpha, delta, output, reinterpret_cast<cudaStream_t>(stream_ptr));
}

std::vector<std::pair<KernelAttr, ApplyGradientDescentKernelMod::LaunchFunc>>
  ApplyGradientDescentKernelMod::func_list_ = {{KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddOutputAttr(kNumberTypeFloat32),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<float>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat16)
                                                  .AddInputAttr(kNumberTypeFloat16)
                                                  .AddInputAttr(kNumberTypeFloat16)
                                                  .AddOutputAttr(kNumberTypeFloat16),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> ApplyGradientDescentKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyGradientDescent, ApplyGradientDescentKernelMod);
}  // namespace kernel
}  // namespace mindspore
