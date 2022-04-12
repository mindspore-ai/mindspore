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

#include "plugin/device/cpu/kernel/dropout_nd_cpu_kernel.h"
#include <algorithm>
#include <random>
#include <utility>
#include <set>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void DropoutNdCpuKernelMod::CheckDropOutNdShape() {
  size_t nd_dims = input_shape_.size();
  size_t expected_dims = 0;
  if (kernel_name_ == prim::kPrimDropout2D->name()) {
    // Dropout2D ---> data format NCHW(4 dims)
    expected_dims = 4;
  } else if (kernel_name_ == prim::kPrimDropout3D->name()) {
    // Dropout3D ---> data format NCDHW(5 dims)
    expected_dims = 5;
  } else {
    MS_LOG(EXCEPTION) << "For 'DropoutNd' should only support Dropout2D or Dropout3D, right now, but got "
                      << kernel_name_;
  }
  if (expected_dims != nd_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " input dims should be " << expected_dims << "D, but got  "
                      << nd_dims << "D.";
  }
}

void DropoutNdCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, kIndex0);
  mask_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, kIndex1);
  CheckDropOutNdShape();
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  const auto keep_prob_attr = "keep_prob";
  if (!common::AnfAlgo::HasNodeAttr(keep_prob_attr, kernel_node)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " has no attribute of  'keep_prob' please check. ";
  }
  keep_prob_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, keep_prob_attr);
  if (keep_prob_ < 0.0 || keep_prob_ > 1.0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the 'keep_prob' should be in [0.0, 1.0], but got " << keep_prob_;
  }
  input_data_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  n_ = input_shape_.at(kDim0);
  c_ = input_shape_.at(kDim1);
  for (const auto &shape : input_shape_) {
    element_nums_ *= shape;
  }
  channels_ = n_ * c_;
  element_per_channel_ = element_nums_ / channels_;

  size_t unit_size = GetTypeByte(TypeIdToType(input_data_dtype_));
  size_t input_size = element_nums_ * unit_size;
  size_t workspace_size = channels_ * sizeof(float);
  size_t mask_output_size = element_nums_ * sizeof(bool);
  input_size_list_.emplace_back(input_size);
  workspace_size_list_.emplace_back(workspace_size);
  output_size_list_.emplace_back(input_size);
  output_size_list_.emplace_back(mask_output_size);
}

template <typename T>
bool DropoutNdCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  const auto input_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  const auto workspace_addr = reinterpret_cast<float *>(workspace.at(kIndex0)->addr);
  auto output_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto mask_addr = reinterpret_cast<bool *>(outputs.at(kIndex1)->addr);

  // When keep_prob equal to 0.0, output default to zero, mask default to false.
  if (keep_prob_ == 0.0) {
    auto ret = memset_s(output_addr, outputs.at(kIndex0)->size, 0, outputs.at(kIndex0)->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " memset_s error.";
    }
    ret = memset_s(mask_addr, outputs.at(kIndex1)->size, 0, outputs.at(kIndex1)->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " memset_s error.";
    }
    return true;
  }

  double scale = 1.f / keep_prob_;
  // Generate random data for every channel
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dis(keep_prob_);
  for (size_t channel = 0; channel < channels_; ++channel) {
    workspace_addr[channel] = static_cast<float>(dis(gen));
  }
  auto task = [this, &input_addr, &workspace_addr, &output_addr, &mask_addr, &scale](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      // Get channel index over all samples
      size_t channel_index = i / element_per_channel_;
      auto drop_f = static_cast<double>(workspace_addr[channel_index] <= keep_prob_);
      output_addr[i] = static_cast<T>(scale * static_cast<double>(input_addr[i]) * drop_f);
      mask_addr[i] = static_cast<bool>(drop_f);
    }
  };
  ParallelLaunchAutoSearch(task, element_nums_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, DropoutNdCpuKernelMod::DropoutNdFunc>> DropoutNdCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   &DropoutNdCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> DropoutNdCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DropoutNdFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Dropout2D, DropoutNdCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Dropout3D, DropoutNdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
