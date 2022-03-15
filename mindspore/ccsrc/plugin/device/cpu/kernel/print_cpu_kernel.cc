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

#include "plugin/device/cpu/kernel/print_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "ir/tensor.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace kernel {
void PrintCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_tensor_num; ++i) {
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
    (void)input_shapes_.emplace_back(input_shape);
    size_t size = input_shape.size() ? 1 : 0;
    for (size_t j = 0; j < input_shape.size(); ++j) {
      size *= input_shape[j];
    }
    (void)input_sizes_.emplace_back(size);
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, PrintFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Print does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool PrintCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs) {
  auto data_type = CheckType<T>();
  if (data_type == kTypeUnknown) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be bool, float, int, or uint, but got unsupported type.";
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (input_sizes_[i] == 0) {
      auto num = reinterpret_cast<T *>(inputs[i]->addr);
      std::cout << *num << std::endl;
    } else {
      ShapeVector shape;
      (void)std::transform(input_shapes_[i].begin(), input_shapes_[i].end(), std::back_inserter(shape),
                           [](const size_t &value) { return SizeToLong(value); });
      Tensor tensor(data_type, shape, inputs[i]->addr, input_sizes_[i] * sizeof(T));
      std::cout << tensor.ToStringNoLimit() << std::endl;
    }
  }
  return true;
}

template <typename T>
TypeId PrintCpuKernelMod::CheckType() {
  if constexpr (std::is_same_v<T, bool>) {
    return kNumberTypeBool;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return kNumberTypeInt8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return kNumberTypeInt16;
  } else if constexpr (std::is_same_v<T, int>) {
    return kNumberTypeInt32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return kNumberTypeInt64;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return kNumberTypeUInt8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return kNumberTypeUInt16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return kNumberTypeUInt32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return kNumberTypeUInt64;
  } else if constexpr (std::is_same_v<T, float16>) {
    return kNumberTypeFloat16;
  } else if constexpr (std::is_same_v<T, float>) {
    return kNumberTypeFloat32;
  } else if constexpr (std::is_same_v<T, double>) {
    return kNumberTypeFloat64;
  }
  return kTypeUnknown;
}

std::vector<std::pair<KernelAttr, PrintCpuKernelMod::PrintFunc>> PrintCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &PrintCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> PrintCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, PrintFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Print, PrintCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
