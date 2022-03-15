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

#include "plugin/device/cpu/kernel/concat_offset_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConcatOffsetOutputNum = 1;
constexpr size_t kConcatOffsetOutputShapeSize = 2;
}  // namespace
void ConcatOffsetCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  cnode_ptr_ = kernel_node;
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  int64_t axis = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  auto input_1_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (axis < 0) {
    axis_ = LongToSize(axis + input_1_shape.size());
  } else {
    axis_ = LongToSize(axis);
  }
  if (axis_ >= input_1_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'axis' should be less than the dimension of 'input_x', but got 'axis': " << axis_
                      << ", and the dimension of 'input_x': " << input_1_shape.size();
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Concat offset does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
}
template <typename T>
bool ConcatOffsetCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConcatOffsetOutputNum, kernel_name_);
  auto node_ = cnode_ptr_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cnode_ptr_(kernel_node) is expired. Error no: " << node_;
  }
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->addr);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node_);
  std::vector<size_t> offset{0};
  size_t all_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0)[axis_];

  // cal offset
  for (size_t i = 1; i < input_num; i++) {
    auto input_shape_i = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, i);
    if (axis_ >= input_shape_i.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'axis' should be less than the dimension of input, but got 'axis': " << axis_
                        << ", and the dimension of the " << i << "'th input: " << input_shape_i.size();
    }
    offset.emplace_back(all_shape);
    all_shape += input_shape_i[axis_];
  }
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node_, 0);
  if (output_shape.size() != kConcatOffsetOutputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output should be "
                      << kConcatOffsetOutputShapeSize << ", but got:" << output_shape.size();
  }
  if (output_shape[0] != input_num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the first dimension value of output should be equal to "
                         "the number of input, but got the first dimension value of output: "
                      << output_shape[0] << ", and the number of input: " << input_num;
  }
  size_t rank = output_shape[1];
  size_t idx = 0;
  for (size_t i = 0; i < input_num; ++i) {
    for (size_t j = 0; j < rank; ++j) {
      if (j == axis_) {
        output_addr[idx] = offset[i];
      } else {
        output_addr[idx] = 0;
      }
      idx++;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, ConcatOffsetCpuKernelMod::ConcatOffsetFunc>> ConcatOffsetCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
   &ConcatOffsetCpuKernelMod::LaunchKernel<bool>}};  // namespace kernel

std::vector<KernelAttr> ConcatOffsetCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, ConcatOffsetFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ConcatOffset, ConcatOffsetCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
