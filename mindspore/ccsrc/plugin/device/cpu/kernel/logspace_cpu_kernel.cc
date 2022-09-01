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

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogSpaceInputsNum = 2;
constexpr size_t kLogSpaceOutputsNum = 1;
}  // namespace
void LogSpaceCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kLogSpaceInputsNum, common::AnfAlgo::GetCNodeName(kernel_node));
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kLogSpaceOutputsNum, common::AnfAlgo::GetCNodeName(kernel_node));
  auto input_shape_1 = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto input_shape_2 = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto input_shape_size_1 = input_shape_1.size();
  auto input_shape_size_2 = input_shape_2.size();
  if (input_shape_size_1 > 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, input[start] must be 0-D, but got " << input_shape_size_1 << "-D.";
  }
  if (input_shape_size_2 > 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, input[end] must be 0-D, but got " << input_shape_size_2 << "-D.";
  }
  steps_ = common::AnfAlgo::GetNodeAttr<size_t>(kernel_node, "steps");
  if (steps_ < 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, attr[steps] must be greater than 0, but got steps: " << steps_ << ".";
  }
  base_ = common::AnfAlgo::GetNodeAttr<size_t>(kernel_node, "base");
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "LogSpace does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T, typename S>
bool LogSpaceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) const {
  auto *input_start_addr = static_cast<T *>(inputs[0]->addr);
  auto *input_end_addr = static_cast<T *>(inputs[1]->addr);
  auto input_start = static_cast<double>(input_start_addr[0]);
  auto input_end = static_cast<double>(input_end_addr[0]);
  auto *output_addr = static_cast<S *>(outputs[0]->addr);
  if (steps_ > 0) {
    double w = (input_end - input_start) / (steps_ - 1);
    double q = pow(base_, w);
    double input_start_value = input_start;
    for (size_t i = 0; i < steps_; i++) {
      double item = pow(base_, input_start_value) * pow(q, i);
      *(output_addr + i) = static_cast<S>(item);
    }
  } else if (steps_ == 1) {
    double w = 1;
    double q = pow(base_, w);
    double input_start_value = input_start;
    for (size_t i = 0; i < steps_; i++) {
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
