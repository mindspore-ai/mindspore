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
#include "mindspore/core/ops/concat_offset.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConcatOffsetOutputNum = 1;
constexpr size_t kConcatOffsetOutputShapeSize = 2;
}  // namespace
bool ConcatOffsetCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConcatOffsetOutputNum, kernel_name_);
  if (inputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", input tensors can not be empty";
    return false;
  }
  auto op_prim = std::dynamic_pointer_cast<ops::ConcatOffset>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  if (op_prim->HasAttr(kAttrAxis)) {
    axis_ = op_prim->get_axis();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "Concat offset does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

int ConcatOffsetCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (output_shape_.size() != kConcatOffsetOutputShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output must be " << kConcatOffsetOutputShapeSize
                  << ", but got:" << output_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  if (LongToSize(output_shape_[kIndex0]) != inputs.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the first dimension value of output must be equal to "
                     "the number of input, but got the first dimension value of output: "
                  << output_shape_[kIndex0] << ", and the number of input: " << inputs.size();
    return KRET_RESIZE_FAILED;
  }
  input_shapes_.clear();
  for (size_t i = 0; i < inputs.size(); i++) {
    ShapeVector shape_i = inputs[i]->GetShapeVector();
    input_shapes_.push_back(shape_i);
    if (shape_i.size() != input_shapes_[kIndex0].size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', input tensors shape's rank must be equal, but got input[0] shape's rank = "
                    << input_shapes_[kIndex0].size() << ", input[" << i << "] shape's rank = " << shape_i.size();
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

template <typename T>
bool ConcatOffsetCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<int64_t *>(outputs[kIndex0]->addr);

  auto x_rank = SizeToLong(input_shapes_[kIndex0].size());
  if (axis_ < -x_rank || axis_ >= x_rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", 'axis' must be in range [-" << x_rank << ", " << x_rank
                      << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += x_rank;
  }
  auto axis = LongToSize(axis_);

  ShapeVector offset{0};
  auto all_shape = input_shapes_[0][axis];

  // cal offset
  for (size_t i = 1; i < inputs.size(); i++) {
    offset.emplace_back(all_shape);
    all_shape += input_shapes_[i][axis];
  }
  size_t rank = LongToSize(output_shape_[kIndex1]);
  size_t idx = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
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
