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

#include "plugin/device/cpu/kernel/hsigmoid_grad_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHSigmoidGradInputsNum = 2;
constexpr size_t kHSigmoidGradOutputsNum = 1;
}  // namespace

std::vector<std::pair<KernelAttr, HSigmoidGradCpuKernelMod::HSigmoidGradFunc>> HSigmoidGradCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &HSigmoidGradCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &HSigmoidGradCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &HSigmoidGradCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &HSigmoidGradCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &HSigmoidGradCpuKernelMod::LaunchKernel<float>}};

void HSigmoidGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  x_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  for (const uint64_t &d : x_shape_) {
    tensor_size_ *= d;
  }

  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, HSigmoidGradFunc> &pair) { return pair.first; });
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "HSigmoidGrad does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool HSigmoidGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHSigmoidGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHSigmoidGradOutputsNum, kernel_name_);
  const auto *dy = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *x = reinterpret_cast<T *>(inputs[1]->addr);
  auto *out = reinterpret_cast<T *>(outputs[0]->addr);

  auto zero = static_cast<T>(0);
  auto three = static_cast<T>(3);
  auto six = static_cast<T>(6);

  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      if (x[i] + three <= zero || x[i] >= three) {
        out[i] = zero;
      } else {
        out[i] = dy[i] / six;
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HSigmoidGrad, HSigmoidGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
