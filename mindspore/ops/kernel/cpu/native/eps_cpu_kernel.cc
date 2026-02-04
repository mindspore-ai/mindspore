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
#include "kernel/cpu/native/eps_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "base/float16.h"

namespace mindspore {
namespace kernel {
namespace eps_cpu {
namespace {
constexpr size_t kEpsInputsNum = 1;
constexpr size_t kEpsOnputsNum = 1;
}  // namespace
bool EpsCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEpsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEpsOnputsNum, kernel_name_);
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
bool EpsCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(output);
  size_t output_size = outputs[0]->size() / sizeof(T);
  T min_val = getEpsilon<T>();
  auto task = [this, output, input, min_val](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = min_val;
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, EpsCpuKernelMod::EpsFunc>> EpsCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &EpsCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &EpsCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &EpsCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> EpsCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EpsFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Eps, EpsCpuKernelMod);
}  // namespace eps_cpu
}  // namespace kernel
}  // namespace mindspore
