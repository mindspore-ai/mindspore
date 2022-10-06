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
#include <vector>
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
  axis_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Concat offset does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
}
template <typename T>
bool ConcatOffsetCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConcatOffsetOutputNum, kernel_name_);
  auto node_ = cnode_ptr_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cnode_ptr_(kernel_node) is expired. Error no: " << node_;
  }
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->addr);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node_);
  if (input_num == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", input tensors can not be empty";
  }
  // check input shapes
  std::vector<ShapeVector> input_shapes;
  for (size_t i = 0; i < input_num; i++) {
    ShapeVector input_shape_i = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, i);
    input_shapes.push_back(input_shape_i);
    if (input_shape_i.size() != input_shapes[0].size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', input tensors shape's rank must be equal, but got input[0] shape's rank = "
                        << input_shapes[0].size() << ", input[" << i << "] shape's rank = " << input_shape_i.size();
    }
  }
  // check axis
  auto x_rank = SizeToLong(input_shapes[0].size());
  if (axis_ < -x_rank || axis_ >= x_rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", 'axis' must be in range [-" << x_rank << ", " << x_rank
                      << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += x_rank;
  }
  auto axis = LongToSize(axis_);

  ShapeVector offset{0};
  auto all_shape = input_shapes[0][axis];

  // cal offset
  for (size_t i = 1; i < input_num; i++) {
    offset.emplace_back(all_shape);
    all_shape += input_shapes[i][axis];
  }
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node_, 0);
  if (output_shape.size() != kConcatOffsetOutputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output must be "
                      << kConcatOffsetOutputShapeSize << ", but got:" << output_shape.size();
  }
  if (LongToSize(output_shape[0]) != input_num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the first dimension value of output must be equal to "
                         "the number of input, but got the first dimension value of output: "
                      << output_shape[0] << ", and the number of input: " << input_num;
  }
  size_t rank = LongToSize(output_shape[1]);
  size_t idx = 0;
  for (size_t i = 0; i < input_num; ++i) {
    for (size_t j = 0; j < rank; ++j) {
      if (j == axis) {
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
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ConcatOffsetFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ConcatOffset, ConcatOffsetCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
