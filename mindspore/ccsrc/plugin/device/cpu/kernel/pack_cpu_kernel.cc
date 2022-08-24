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

#include "plugin/device/cpu/kernel/pack_cpu_kernel.h"
#include <thread>
#include <algorithm>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPackOutputsNum = 1;
}  // namespace

void PackFwdCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
  axis_ = LongToInt(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  if (axis_ < 0) {
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    axis_ += (SizeToInt(input_shape.size()) + 1);
  }

  dims_behind_axis_ = 1;
  // calculate elements while dim >= axis
  auto first_input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  for (size_t i = IntToSize(axis_); i < first_input_shape.size(); i++) {
    dims_behind_axis_ *= static_cast<size_t>(first_input_shape[i]);
  }

  auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_size_ *= static_cast<size_t>(output_shape[i]);
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool PackFwdCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto node_ = cnode_ptr_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cnode_ptr_(kernel_node) is expired. Error no: " << node_;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPackOutputsNum, kernel_name_);
  auto *output = reinterpret_cast<char *>(outputs[0]->addr);
  std::vector<char *> inputs_host;
  for (size_t i = 0; i < inputs.size(); i++) {
    (void)inputs_host.emplace_back(reinterpret_cast<char *>(inputs[i]->addr));
  }

  // multi-threading
  size_t input_size = output_size_;
  size_t dims_behind_axis = dims_behind_axis_;
  size_t copy_time = input_size / dims_behind_axis;
  size_t single_copy_size = dims_behind_axis * sizeof(T);
  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      size_t cur_input_index = pos % this->input_num_;
      size_t local_idx = pos / this->input_num_;
      (void)memcpy_s(output + single_copy_size * pos, single_copy_size,
                     inputs_host[cur_input_index] + single_copy_size * local_idx, single_copy_size);
    }
  };
  ParallelLaunchAutoSearch(task, copy_time, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, PackFwdCpuKernelMod::PackFunc>> PackFwdCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &PackFwdCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &PackFwdCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &PackFwdCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &PackFwdCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &PackFwdCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &PackFwdCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &PackFwdCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &PackFwdCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &PackFwdCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &PackFwdCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &PackFwdCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &PackFwdCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &PackFwdCpuKernelMod::LaunchKernel<complex128>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &PackFwdCpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> PackFwdCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PackFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Stack, PackFwdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
