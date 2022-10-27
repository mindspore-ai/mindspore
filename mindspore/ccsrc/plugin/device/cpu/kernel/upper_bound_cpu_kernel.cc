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
#include "plugin/device/cpu/kernel/upper_bound_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool UpperBoundCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpperBoundCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  sorted_x_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  values_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();

  size_t size_exp = 2;
  if (sorted_x_shape_.size() != values_shape_.size() || sorted_x_shape_.size() != size_exp ||
      sorted_x_shape_[0] != values_shape_[0]) {
    MS_LOG(ERROR) << "The shape of input is invalid.";
    return KRET_RESIZE_FAILED;
  }
  sorted_x_num_ = static_cast<size_t>(sorted_x_shape_[0] * sorted_x_shape_[1]);
  values_num_ = static_cast<size_t>(values_shape_[0] * values_shape_[1]);
  output_num_ = static_cast<size_t>(output_shape_[0] * output_shape_[1]);
  if (values_num_ != output_num_) {
    MS_LOG(ERROR) << "Infer the shape of output error.";
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

template <typename I, typename O>
bool UpperBoundCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto sorted_x_data_addr = static_cast<I *>(inputs[0]->addr);
  auto values_data_addr = static_cast<I *>(inputs[1]->addr);
  auto output_data_addr = static_cast<O *>(outputs[0]->addr);
  size_t sorted_x_data_column = static_cast<size_t>(sorted_x_shape_[1]);
  size_t values_data_column = static_cast<size_t>(values_shape_[1]);
  auto task = [this, &values_data_addr, &sorted_x_data_addr, &output_data_addr, &sorted_x_data_column,
               &values_data_column](size_t start, size_t end) {
    const size_t kNumber2 = 2;
    for (size_t i = 0; i < this->values_num_; i++) {
      size_t seq_row = i / values_data_column;
      int64_t low = seq_row * sorted_x_data_column;
      int64_t up = (seq_row + 1) * sorted_x_data_column - 1;
      while (low <= up) {
        int64_t mid = (low + up) / kNumber2;
        if (values_data_addr[i] < sorted_x_data_addr[static_cast<size_t>(mid)]) {
          up = mid - 1;
        } else {
          low = mid + 1;
        }
      }
      output_data_addr[i] = static_cast<O>(low - SizeToLong(seq_row) * SizeToLong(sorted_x_data_column));
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

std::vector<std::pair<KernelAttr, UpperBoundCpuKernelMod::UpperBoundFunc>> UpperBoundCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundCpuKernelMod::LaunchKernel<uint16_t, int64_t>}};

std::vector<KernelAttr> UpperBoundCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpperBoundFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpperBound, UpperBoundCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
