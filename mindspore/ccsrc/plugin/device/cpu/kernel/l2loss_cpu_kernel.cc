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

#include "plugin/device/cpu/kernel/l2loss_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kL2LossInputsNum = 1;
constexpr size_t kL2LossOutputsNum = 1;
}  // namespace

void L2LossCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  tensor_size_ = 1;
  for (const size_t &d : input_shape_) {
    tensor_size_ *= d;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, L2LossFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "L2Loss does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool L2LossCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kL2LossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kL2LossOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  *result_addr = static_cast<T>(0);
  if (tensor_size_ == 0) {
    MS_LOG(WARNING) << kernel_name_ << " input shape contain 0, input_shape: " << input_shape_;
    return true;
  }
  for (size_t i = 0; i < tensor_size_; i++) {
    *result_addr += input_addr[i] * input_addr[i];
  }
  *result_addr = *result_addr / 2;
  return true;
}

std::vector<std::pair<KernelAttr, L2LossCpuKernelMod::L2LossFunc>> L2LossCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &L2LossCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &L2LossCpuKernelMod::LaunchKernel<float>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, L2Loss, L2LossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
