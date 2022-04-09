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
#include "plugin/device/cpu/kernel/lerp_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
namespace mindspore {
namespace kernel {
void LerpCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  start_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  end_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex1);
  weight_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex2);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, kIndex0);
  for (const auto &out_shape : output_shape_) {
    output_size_ *= out_shape;
  }
}

template <typename T>
bool LerpCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (start_shape_ == end_shape_ && start_shape_ == weight_shape_) {
    auto *input_start = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
    auto *input_end = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
    auto *input_weight = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
    T *output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
    auto task = [&input_start, &input_end, &input_weight, &output](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T start_value = input_start[i];
        T end_value = input_end[i];
        T weight_value = input_weight[i];
        output[i] = static_cast<T>(start_value + (end_value - start_value) * weight_value);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else {
    MultipleBroadcastIterator multi_broadcast_iterator({start_shape_, end_shape_, weight_shape_}, output_shape_);
    auto *input_start = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
    auto *input_end = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
    auto *input_weight = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
    T *output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
    auto task = [&input_start, &input_end, &input_weight, &output, &multi_broadcast_iterator](size_t start,
                                                                                              size_t end) {
      auto iter = multi_broadcast_iterator;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        T start_value = input_start[iter.GetInputPos(kIndex0)];
        T end_value = input_end[iter.GetInputPos(kIndex1)];
        T weight_value = input_weight[iter.GetInputPos(kIndex2)];
        output[i] = static_cast<T>(start_value + (end_value - start_value) * weight_value);
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, LerpCpuKernelMod::LerpFunc>> LerpCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddAllSameAttr(true)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LerpCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddAllSameAttr(true)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LerpCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> LerpCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LerpFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Lerp, LerpCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
