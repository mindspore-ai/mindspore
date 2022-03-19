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

#include "plugin/device/cpu/kernel/masked_fill_cpu_kernel.h"
#include <algorithm>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedFillInputsNum = 3;
constexpr size_t kMaskedFillOutputsNum = 1;
}  // namespace

void MaskedFillCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto mask_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_shape != mask_shape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input shape should equal to mask shape, but got input shape: " << input_shape
                      << ", mask shape: " << mask_shape;
  }
  shape_size_ = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
    shape_size_ *= input_shape[i];
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MaskedFill does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool MaskedFillCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedFillInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedFillOutputsNum, kernel_name_);

  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto mask = reinterpret_cast<bool *>(inputs[1]->addr);
  auto value = reinterpret_cast<T *>(inputs[2]->addr)[0];
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  auto task = [input, mask, output, value](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = mask[i] ? value : input[i];
    }
  };

  ParallelLaunchAutoSearch(task, shape_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, MaskedFillCpuKernelMod::MaskedFillFunc>> MaskedFillCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &MaskedFillCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &MaskedFillCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &MaskedFillCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MaskedFillCpuKernelMod::LaunchKernel<int32_t>},
};

std::vector<KernelAttr> MaskedFillCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaskedFillFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaskedFill, MaskedFillCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
