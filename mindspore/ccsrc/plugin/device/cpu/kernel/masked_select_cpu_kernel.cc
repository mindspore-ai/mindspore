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

#include "plugin/device/cpu/kernel/masked_select_cpu_kernel.h"
#include <algorithm>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedSelectInputsNum = 2;
constexpr size_t kMaskedSelectOutputsNum = 1;
}  // namespace

void MaskedSelectCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_a_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_shape_b_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = CPUKernelUtils::GetBroadcastShape(input_shape_a_, input_shape_b_);
  for (const uint64_t &d : output_shape_) {
    tensor_size_ *= d;
  }
  node_wpt_ = kernel_node;

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MaskedSelect does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  is_need_updateop_ = true;
}

template <typename T>
bool MaskedSelectCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedSelectInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedSelectOutputsNum, kernel_name_);
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto mask = reinterpret_cast<bool *>(inputs[1]->addr);
  auto y = reinterpret_cast<T *>(outputs[0]->addr);
  uint64_t j = 0;
  if (input_shape_a_ == input_shape_b_) {
    for (uint64_t i = 0; i < tensor_size_; ++i) {
      if (mask[i]) {
        y[j++] = x[i];
      }
    }
  } else {  // Broadcast
    BroadcastIterator iter(input_shape_a_, input_shape_b_, output_shape_);
    iter.SetPos(0);
    for (uint64_t i = 0; i < tensor_size_; ++i) {
      if (mask[iter.GetInputPosB()]) {
        y[j++] = x[iter.GetInputPosA()];
      }
      iter.GenNextPos();
    }
  }
  if (!node_wpt_.expired()) {
    auto node_ = node_wpt_.lock();
    if (!node_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node_;
    }
    std::vector<size_t> out_shape;
    (void)out_shape.emplace_back(j);
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(node_);
    std::vector<TypeId> dtypes(output_num);
    for (size_t i = 0; i < output_num; i++) {
      dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node_, i);
    }
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape}, node_.get());
  }
  return true;
}

std::vector<std::pair<KernelAttr, MaskedSelectCpuKernelMod::MaskedSelectFunc>> MaskedSelectCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),
   &MaskedSelectCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
   &MaskedSelectCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt16),
   &MaskedSelectCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
   &MaskedSelectCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16),
   &MaskedSelectCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64),
   &MaskedSelectCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MaskedSelectCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaskedSelectFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaskedSelect, MaskedSelectCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
