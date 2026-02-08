/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "kernel/gpu/cuda/math/eps_gpu_kernel.h"

#include <algorithm>
#include <limits>

#include "base/float16.h"
#include "kernel/gpu/cuda_impl/cuda_ops/eps_impl.cuh"

namespace mindspore {
namespace kernel {
bool EpsGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size();
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << outputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the supported data types are ['float16', 'float32', 'float64'], but got: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int EpsGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
  output_elements_ = SizeOf(outputs[0]->GetShapeVector());
  return KRET_OK;
}

template <typename T>
T getEpsilon() {
  T epsilon = static_cast<T>(0.5);
  T one = static_cast<T>(1.0);
  T two = static_cast<T>(2.0);
  while (one + epsilon / two > one) {
    epsilon = epsilon / two;
  }
  return epsilon;
}

template <typename T>
bool EpsGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  T *output = reinterpret_cast<T *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(output);
  T min_val = getEpsilon<T>();
  CalEps(output_elements_, input, output, min_val, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, EpsGpuKernelMod::EpsFunc>> EpsGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &EpsGpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &EpsGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &EpsGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> EpsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EpsFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Eps, EpsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
