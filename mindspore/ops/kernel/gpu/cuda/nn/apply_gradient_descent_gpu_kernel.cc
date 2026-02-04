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

#include "kernel/gpu/cuda/nn/apply_gradient_descent_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ApplyGradientDescentKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                 const std::vector<kernel::KernelTensor *> &outputs, void *stream_ptr) {
  T *var = GetDeviceAddress<T>(inputs, kIndex0);
  T *alpha = GetDeviceAddress<T>(inputs, kIndex1);
  T *delta = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto status =
    CalApplyGradientDescent(input_size_, var, alpha, delta, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
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
                                                &ApplyGradientDescentKernelMod::LaunchKernel<half>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeInt8)
                                                  .AddInputAttr(kNumberTypeInt8)
                                                  .AddInputAttr(kNumberTypeInt8)
                                                  .AddOutputAttr(kNumberTypeInt8),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<int8_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeUInt8)
                                                  .AddInputAttr(kNumberTypeUInt8)
                                                  .AddInputAttr(kNumberTypeUInt8)
                                                  .AddOutputAttr(kNumberTypeUInt8),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<uint8_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeInt16)
                                                  .AddInputAttr(kNumberTypeInt16)
                                                  .AddInputAttr(kNumberTypeInt16)
                                                  .AddOutputAttr(kNumberTypeInt16),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<int16_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeUInt16)
                                                  .AddInputAttr(kNumberTypeUInt16)
                                                  .AddInputAttr(kNumberTypeUInt16)
                                                  .AddOutputAttr(kNumberTypeUInt16),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<uint16_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeUInt32)
                                                  .AddInputAttr(kNumberTypeUInt32)
                                                  .AddInputAttr(kNumberTypeUInt32)
                                                  .AddOutputAttr(kNumberTypeUInt32),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<uint32_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeInt64)
                                                  .AddInputAttr(kNumberTypeInt64)
                                                  .AddInputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<int64_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeUInt64)
                                                  .AddInputAttr(kNumberTypeUInt64)
                                                  .AddInputAttr(kNumberTypeUInt64)
                                                  .AddOutputAttr(kNumberTypeUInt64),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<uint64_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat64)
                                                  .AddInputAttr(kNumberTypeFloat64)
                                                  .AddInputAttr(kNumberTypeFloat64)
                                                  .AddOutputAttr(kNumberTypeFloat64),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<double>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeComplex64)
                                                  .AddInputAttr(kNumberTypeComplex64)
                                                  .AddInputAttr(kNumberTypeComplex64)
                                                  .AddOutputAttr(kNumberTypeComplex64),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<utils::Complex<float>>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeComplex128)
                                                  .AddInputAttr(kNumberTypeComplex128)
                                                  .AddInputAttr(kNumberTypeComplex128)
                                                  .AddOutputAttr(kNumberTypeComplex128),
                                                &ApplyGradientDescentKernelMod::LaunchKernel<utils::Complex<double>>}};

std::vector<KernelAttr> ApplyGradientDescentKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyGradientDescent, ApplyGradientDescentKernelMod);
}  // namespace kernel
}  // namespace mindspore
