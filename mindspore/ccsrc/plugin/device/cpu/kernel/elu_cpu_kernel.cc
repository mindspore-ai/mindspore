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

#include "plugin/device/cpu/kernel/elu_cpu_kernel.h"
#include "nnacl/fp32/activation_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
bool EluCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs) {
  const auto *in = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *out = reinterpret_cast<T *>(outputs[0]->device_ptr());
  const size_t lens = outputs[0]->size() / sizeof(T);
  auto alpha = inputs[kIndex1]->GetValueWithCheck<double>();

  auto task = [in, out, alpha](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      (void)::Elu(in + start, SizeToInt(end - start), out + start, alpha);
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = in[i] > 0 ? in[i] : (std::expm1(in[i]) * alpha);
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, EluCpuKernelMod::KernelRunFunc>> &EluCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, EluCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat32),
     &EluCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &EluCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Elu, EluCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
