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

#include "plugin/device/cpu/kernel/heaviside_cpu_kernel.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kHeavisideInputsNum = 2;
const size_t kHeavisideOutputsNum = 1;
}  // namespace

bool HeavisideCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  input0_shape = inputs[0]->GetShapeVector();
  input1_shape = inputs[1]->GetShapeVector();
  output_shape = outputs[0]->GetShapeVector();
  input0_dtype_ = inputs[0]->GetDtype();
  input1_dtype_ = inputs[1]->GetDtype();
  if (input0_dtype_ != input1_dtype_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'x' and 'values' should have the same data "
                         "type, but got the dtype of 'x': "
                      << input0_dtype_ << " and the dtype of 'values': " << input1_dtype_ << ".";
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool HeavisideCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHeavisideInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHeavisideOutputsNum, kernel_name_);

  BroadcastIterator base_iter(input0_shape, input1_shape, output_shape);
  const T *input0 = static_cast<const T *>(inputs[0]->addr);
  const T *input1 = static_cast<const T *>(inputs[1]->addr);
  auto *output = static_cast<T *>(outputs[0]->addr);
  auto task = [this, &input0, &input1, &output, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      output[i] = static_cast<double>(0) == static_cast<double>(input0[iter.GetInputPosA()])
                    ? input1[iter.GetInputPosB()]
                    : static_cast<T>(input0[iter.GetInputPosA()] > static_cast<T>(0));
      iter.GenNextPos();
    }
  };
  size_t output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); ++i) {
    output_size_ *= static_cast<size_t>(output_shape[i]);
  }
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, HeavisideCpuKernelMod::HeavisideLaunchFunc>> HeavisideCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &HeavisideCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &HeavisideCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &HeavisideCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &HeavisideCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &HeavisideCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &HeavisideCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &HeavisideCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &HeavisideCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &HeavisideCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &HeavisideCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &HeavisideCpuKernelMod::LaunchKernel<uint64_t>}};

std::vector<KernelAttr> HeavisideCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HeavisideLaunchFunc> &pair) { return pair.first; });

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Heaviside, HeavisideCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
