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

#include "plugin/device/cpu/kernel/concat_offset_v1_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConcatOffsetV1AxisNum = 1;
}  // namespace
void ConcatOffsetV1CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  cnode_ptr_ = kernel_node;
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "ConcatOffsetV1 does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool ConcatOffsetV1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), inputs.size() - kConcatOffsetV1AxisNum, kernel_name_);
  auto node_ = cnode_ptr_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cnode_ptr_(kernel_node) is expired. Error no: " << node_;
  }

  auto axis = static_cast<int64_t>(*reinterpret_cast<int32_t *>(inputs[0]->addr));
  auto input_0_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 1);
  int64_t input_0_elem_num = input_0_shape[0];
  if (axis >= input_0_elem_num || axis < -input_0_elem_num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be fall in range [-" << input_0_elem_num << ", "
                      << input_0_elem_num << "), but got 'axis': " << axis;
  }
  if (axis < 0) {
    axis_ = LongToSize(axis + input_0_elem_num);
  } else {
    axis_ = LongToSize(axis);
  }

  size_t input_tensor_num = common::AnfAlgo::GetInputTensorNum(node_) - kConcatOffsetV1AxisNum;
  size_t output_tensor_num = common::AnfAlgo::GetOutputTensorNum(node_);
  if (output_tensor_num != input_tensor_num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the output tensor num must be equal to "
                         "the input x tensor num, but got the output tensor num: "
                      << output_tensor_num << ", and the input x tensor num: " << input_tensor_num;
  }

  auto output_shape = common::AnfAlgo::GetOutputInferShape(node_, 0);
  size_t elem_num = LongToSize(output_shape[0]);
  int32_t offset = 0;
  auto input0_addr = reinterpret_cast<int32_t *>(inputs[1]->addr);
  for (size_t i = 0; i < input_tensor_num; ++i) {
    auto input_i_addr = reinterpret_cast<int32_t *>(inputs[i + 1]->addr);
    auto output_i_addr = reinterpret_cast<int32_t *>(outputs[i]->addr);
    for (size_t j = 0; j < elem_num; ++j) {
      if (j == axis_) {
        output_i_addr[j] = offset;
        offset += input_i_addr[j];
      } else {
        if (input_i_addr[j] != input0_addr[j]) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', except for the " << axis_
                            << "th axis, all elements in other axes should be equal,"
                               " but for the "
                            << j << "th axis, element in input x" << i << " is " << input_i_addr[j]
                            << ", and element in input x0 is " << input0_addr[j];
        }
        output_i_addr[j] = 0;
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, ConcatOffsetV1CpuKernelMod::ConcatOffsetV1Func>>
  ConcatOffsetV1CpuKernelMod::func_list_ = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ConcatOffsetV1CpuKernelMod::LaunchKernel<int32_t>}};

std::vector<KernelAttr> ConcatOffsetV1CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ConcatOffsetV1Func> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ConcatOffsetV1, ConcatOffsetV1CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
