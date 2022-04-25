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

#include "plugin/device/gpu/kernel/math/cum_op_gpu_kernel.h"
#include <functional>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kCumInputsNum = 1;
constexpr int kCumOutputsNum = 2;
constexpr char AXIS[] = "axis";

static const std::map<std::string, CumOpType> kCumOpTypeMap = {
  {"Cummin", CUMMIN},
};
}  // namespace

void CumOpGpuKernelMod::ResetResource() noexcept {
  inner_size_ = 1;
  outer_size_ = 1;
  axis_size_ = 1;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool CumOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << ", but got kernel name as " << kernel_name_;
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }

  auto iter = kCumOpTypeMap.find(kernel_name_);
  if (iter == kCumOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Only support these cum operators: " << Map2Str(kCumOpTypeMap) << " currently, but got "
                      << kernel_name_;
  }
  cum_op_type_ = iter->second;
  t_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  s_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex1).first);
  kernel_func_ = func_list_[index].second;
  return true;
}

bool CumOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  std::vector<int64_t> input_shape = inputs[kIndex0]->GetShapeVector();
  auto rank = SizeToLong(input_shape.size());
  auto axis_input = GetValue<int64_t>(base_operator->GetAttr(AXIS));
  auto axis = axis_input < 0 ? LongToSize(axis_input + rank) : LongToSize(axis_input);
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i < axis) {
      outer_size_ *= input_shape.at(i);
    } else if (i > axis) {
      inner_size_ *= input_shape.at(i);
    } else {
      axis_size_ = input_shape.at(i);
    }
  }

  element_size_ = outer_size_ * inner_size_ * axis_size_;
  if (!element_size_) {
    return true;
  }

  input_size_list_.push_back(element_size_ * t_size_);
  output_size_list_.push_back(element_size_ * t_size_);
  output_size_list_.push_back(element_size_ * s_size_);
  workspace_size_list_.push_back(element_size_ * sizeof(size_t));
  return true;
}

template <typename T, typename S>
bool CumOpGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (!element_size_) {
    return true;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto input_ptr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto value_ptr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto index_ptr = reinterpret_cast<S *>(outputs.at(kIndex1)->addr);
  auto workspace_ptr = reinterpret_cast<size_t *>(workspace.at(kIndex0)->addr);

  CumOp(cum_op_type_, input_ptr, workspace_ptr, value_ptr, index_ptr, element_size_, axis_size_, inner_size_,
        cuda_stream);
  return true;
}

std::vector<std::pair<KernelAttr, CumOpGpuKernelMod::CumOpLaunchFunc>> CumOpGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<half, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &CumOpGpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &CumOpGpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> CumOpGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumOpGpuKernelMod::CumOpLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(NativeGpuKernelMod, Cummin, CumOpGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
