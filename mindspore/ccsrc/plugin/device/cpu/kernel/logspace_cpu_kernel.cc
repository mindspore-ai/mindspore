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

#include "plugin/device/cpu/kernel/logspace_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/log_space.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogSpaceInputsNum = 2;
constexpr size_t kLogSpaceOutputsNum = 1;
}  // namespace

bool LogSpaceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLogSpaceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLogSpaceOutputsNum, kernel_name_);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::LogSpace>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  steps_ = kernel_ptr->get_steps();
  if (steps_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', attr[steps] must be greater than 0, but got steps: " << steps_
                  << ".";
    return false;
  }
  base_ = kernel_ptr->get_base();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LogSpaceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_shape_1 = inputs[kIndex0]->GetShapeVector();
  auto input_shape_2 = inputs[kIndex1]->GetShapeVector();
  auto input_shape_size_1 = input_shape_1.size();
  auto input_shape_size_2 = input_shape_2.size();
  if (input_shape_size_1 > 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[start] must be 0-D, but got " << input_shape_size_1 << "-D.";
    return KRET_RESIZE_FAILED;
  }
  if (input_shape_size_2 > 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[end] must be 0-D, but got " << input_shape_size_2 << "-D.";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool LogSpaceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto *input_start_addr = static_cast<T *>(inputs[0]->addr);
  auto *input_end_addr = static_cast<T *>(inputs[1]->addr);
  auto input_start = static_cast<double>(input_start_addr[0]);
  auto input_end = static_cast<double>(input_end_addr[0]);
  auto *output_addr = static_cast<S *>(outputs[0]->addr);
  if (steps_ > 0) {
    double w = (input_end - input_start) / (steps_ - 1);
    double q = pow(base_, w);
    double input_start_value = input_start;
    for (int64_t i = 0; i < steps_; i++) {
      double item = pow(base_, input_start_value) * pow(q, i);
      *(output_addr + i) = static_cast<S>(item);
    }
  } else if (steps_ == 1) {
    double w = 1;
    double q = pow(base_, w);
    double input_start_value = input_start;
    for (int64_t i = 0; i < steps_; i++) {
      double item = pow(base_, input_start_value) * pow(q, i);
      *(output_addr + i) = static_cast<S>(item);
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, LogSpaceCpuKernelMod::LogSpaceFunc>> LogSpaceCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &LogSpaceCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
   &LogSpaceCpuKernelMod::LaunchKernel<float, float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &LogSpaceCpuKernelMod::LaunchKernel<float16, float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
   &LogSpaceCpuKernelMod::LaunchKernel<float16, float>}};

std::vector<KernelAttr> LogSpaceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LogSpaceFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogSpace, LogSpaceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
