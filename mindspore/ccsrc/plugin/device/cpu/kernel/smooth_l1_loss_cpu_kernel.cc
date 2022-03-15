/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/smooth_l1_loss_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSmoothL1LossInputsNum = 2;
constexpr size_t kSmoothL1LossOutputsNum = 1;
}  // namespace
void SmoothL1LossCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  beta_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "beta");
  if (beta_ == 0.0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the 'beta' should not be 0.";
  }
  std::vector<size_t> x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape) {
    tensor_size_ *= d;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SmoothL1Loss does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool SmoothL1LossCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSmoothL1LossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSmoothL1LossOutputsNum, kernel_name_);
  const auto *predict_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *target_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T zero = (T)0.0;
  T half = (T)0.5;
  T beta = (T)beta_;
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      T diff = predict_addr[i] - target_addr[i];
      if (diff < zero) {
        diff = -diff;
      }
      if (diff < beta) {
        result_addr[i] = half * diff * diff / beta;
      } else {
        result_addr[i] = diff - (half * beta);
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, SmoothL1LossCpuKernelMod::SmoothL1LossFunc>> SmoothL1LossCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SmoothL1LossCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SmoothL1LossCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> SmoothL1LossCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, SmoothL1LossFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SmoothL1Loss, SmoothL1LossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
