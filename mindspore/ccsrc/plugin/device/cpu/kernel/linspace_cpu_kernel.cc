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

#include "plugin/device/cpu/kernel/linspace_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kInputNum = 3;
constexpr auto kOutputNum = 1;
}  // namespace

void LinSpaceCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
  }

  auto start = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto end = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto value_count = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);

  // error checking input data
  if ((start.size() != 0) || (end.size() != 0)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', both start and end should be 0-D Tensors, but got dimension "
                      << "of start: " << start.size() << " and dimension of end: " << end.size();
  }

  if (value_count.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of output should be 1, but got "
                      << value_count.size();
  }
  value_count_ = value_count[0];

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SolveTriangular does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool LinSpaceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto value_start = *reinterpret_cast<T *>(inputs[0]->addr);
  auto value_stop = *reinterpret_cast<T *>(inputs[1]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  T add_value = ((value_stop - value_start) / (value_count_ - 1));

  auto task = [output, value_start, add_value](size_t start, size_t end) {
    for (size_t i = 0; i < end; i++) {
      output[i] = value_start + add_value * i;
    }
  };

  ParallelLaunchAutoSearch(task, value_count_, this, &parallel_search_info_);

  return true;
}

std::vector<std::pair<KernelAttr, LinSpaceCpuKernelMod::LinSpaceFunc>> LinSpaceCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &LinSpaceCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LinSpaceCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LinSpaceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LinSpaceFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LinSpace, LinSpaceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
