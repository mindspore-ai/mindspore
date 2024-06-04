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

#include "plugin/device/cpu/kernel/elu_grad_cpu_kernel.h"
#include <cmath>
#include <string>
#include <thread>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
bool EluGradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs) {
  const auto *input0 = reinterpret_cast<T *>(inputs[0]->device_ptr());
  const auto *input1 = reinterpret_cast<T *>(inputs[1]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[0]->device_ptr());

  size_t lens = outputs[0]->size() > 0 ? static_cast<size_t>(outputs[0]->size() / sizeof(T)) : 1;
  auto task = [input0, input1, output](const size_t start, const size_t end) {
    const T alpha = T(1);
    for (size_t i = start; i < end; i++) {
      output[i] = (input1[i] < static_cast<T>(0)) ? input0[i] * (input1[i] + alpha) : input0[i];
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, EluGradCpuKernelMod::KernelRunFunc>> &EluGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, EluGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &EluGradCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat).AddInputAttr(kNumberTypeFloat).AddOutputAttr(kNumberTypeFloat),
     &EluGradCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &EluGradCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EluGrad, EluGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
