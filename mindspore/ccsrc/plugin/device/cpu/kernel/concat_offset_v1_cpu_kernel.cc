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
#include "mindspore/core/utils/check_convert_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConcatOffsetV1AxisNum = 1;
constexpr int64_t kInputMinNumber = 2;
constexpr auto kInputStr = "input number";
}  // namespace
bool ConcatOffsetV1CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  (void)CheckAndConvertUtils::CheckInteger(kInputStr, SizeToLong(inputs.size()), kGreaterEqual, kInputMinNumber,
                                           kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), inputs.size() - kConcatOffsetV1AxisNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "ConcatOffsetV1 does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ConcatOffsetV1CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input0_ = inputs[kIndex1]->GetShapeVector();
  output_ = outputs[kIndex0]->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool ConcatOffsetV1CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto axis = static_cast<int64_t>(*reinterpret_cast<int32_t *>(inputs[kIndex0]->addr));
  int64_t input_0_elem_num = input0_[0];
  if (axis >= input_0_elem_num || axis < -input_0_elem_num) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' must be fall in range [-" << input_0_elem_num << ", "
                  << input_0_elem_num << "), but got 'axis': " << axis;
    return false;
  }
  if (axis < 0) {
    axis_ = LongToSize(axis + input_0_elem_num);
  } else {
    axis_ = LongToSize(axis);
  }

  size_t input_tensor_num = inputs.size() - kConcatOffsetV1AxisNum;
  size_t elem_num = LongToSize(output_[kIndex0]);
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
          MS_LOG(ERROR) << "For '" << kernel_name_ << "', except for the " << axis_
                        << "th axis, all elements in other axes should be equal,"
                           " but for the "
                        << j << "th axis, element in input x" << i << " is " << input_i_addr[j]
                        << ", and element in input x0 is " << input0_addr[j];
          return false;
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
