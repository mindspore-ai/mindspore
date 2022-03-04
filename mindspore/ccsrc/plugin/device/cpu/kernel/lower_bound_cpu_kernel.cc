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
#include "plugin/device/cpu/kernel/lower_bound_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void LowerBoundCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  sorted_x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  values_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  size_t size_exp = 2;
  if (sorted_x_shape_.size() != values_shape_.size() || sorted_x_shape_.size() != size_exp ||
      sorted_x_shape_[0] != values_shape_[0]) {
    MS_LOG(EXCEPTION) << "The shape of input is invalid.";
  }
  sorted_x_num_ = sorted_x_shape_[0] * sorted_x_shape_[1];
  values_num_ = values_shape_[0] * values_shape_[1];
  output_num_ = output_shape_[0] * output_shape_[1];
  if (values_num_ != output_num_) {
    MS_LOG(EXCEPTION) << "Infer the shape of output error.";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, LowerBoundFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "LowerBound does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename I, typename O>
bool LowerBoundCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto sorted_x_data_addr = reinterpret_cast<I *>(inputs[0]->addr);
  auto values_data_addr = reinterpret_cast<I *>(inputs[1]->addr);
  auto output_data_addr = reinterpret_cast<O *>(outputs[0]->addr);
  size_t sorted_x_data_column = sorted_x_shape_[1];
  size_t values_data_column = values_shape_[1];
  auto task = [&](size_t start, size_t end) {
    for (size_t i = 0; i < values_num_; i++) {
      size_t seq_row = i / values_data_column;
      size_t low = seq_row * sorted_x_data_column;
      size_t up = (seq_row + 1) * sorted_x_data_column - 1;
      while (low <= up) {
        size_t mid = (low + up) / 2;
        if (values_data_addr[i] <= sorted_x_data_addr[mid]) {
          up = mid - 1;
        } else {
          low = mid + 1;
        }
      }
      output_data_addr[i] = static_cast<O>(low - seq_row * sorted_x_data_column);
    }
  };
  const size_t kDataSizeThreshold_ = 4 * 1024;
  if (values_num_ * sizeof(I) < kDataSizeThreshold_) {
    task(0, values_num_);
  } else {
    CPUKernelUtils::ParallelFor(task, values_num_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, LowerBoundCpuKernelMod::LowerBoundFunc>> LowerBoundCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   &LowerBoundCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &LowerBoundCpuKernelMod::LaunchKernel<uint16_t, int64_t>}};

std::vector<KernelAttr> LowerBoundCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, LowerBoundFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LowerBound, LowerBoundCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
