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
#include "plugin/device/cpu/kernel/selu_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
namespace mindspore {
namespace kernel {
void SeluCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, kIndex0);
  if (input_shape_ != output_shape_) {
    MS_LOG(EXCEPTION) << kernel_name_ << " input shape does not match to output_shape " << input_shape_ << " vs "
                      << output_shape_;
  }
  for (const auto &out_shape : output_shape_) {
    output_size_ *= out_shape;
  }
}

template <typename T>
bool SeluCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  // The below alpha value and scale value is predefined, according to https://arxiv.org/abs/1706.02515
  double alpha = 1.67326324;
  double scale = 1.05070098;
  auto *input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  T *output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto task = [&input, &output, &alpha, &scale](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto input_value = static_cast<double>(input[i]);
      output[i] = static_cast<T>(input_value >= 0.0 ? scale * input_value : scale * alpha * std::expm1(input_value));
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, SeluCpuKernelMod::SeluFunc>> SeluCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SeluCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SeluCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> SeluCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SeluFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SeLU, SeluCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
