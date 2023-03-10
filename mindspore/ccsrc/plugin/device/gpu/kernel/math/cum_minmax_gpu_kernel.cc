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

#include "plugin/device/gpu/kernel/math/cum_minmax_gpu_kernel.h"
#include <functional>
#include <algorithm>
#include "kernel/common_utils.h"
#include "mindspore/core/ops/cummin.h"
#include "mindspore/core/ops/cummax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kCumInputsNum = 1;
constexpr int kCumOutputsNum = 2;
}  // namespace

bool CumMinMaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[cum_op_type_][index].second;
  switch (cum_op_type_) {
    case CUMMIN: {
      auto kernel_ptr = std::make_shared<ops::Cummin>(base_operator->GetPrim());
      axis_ = kernel_ptr->get_axis();
      break;
    }
    case CUMMAX: {
      auto kernel_ptr = std::make_shared<ops::Cummax>(base_operator->GetPrim());
      axis_ = kernel_ptr->get_axis();
      break;
    }
    default: {
      MS_LOG(ERROR) << "CumMin/CumMax Something unexpected happened!";
      return false;
    }
  }
  return true;
}

int CumMinMaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  outer_size_ = inner_size_ = axis_size_ = 1;
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto rank = SizeToLong(input_shape.size());
  auto axis = axis_ < 0 ? LongToSize(axis_ + rank) : LongToSize(axis_);
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i < axis) {
      outer_size_ *= input_shape.at(i);
    } else if (i > axis) {
      inner_size_ *= input_shape.at(i);
    } else {
      axis_size_ = input_shape.at(i);
    }
  }
  return 0;
}

template <typename DataType, typename IndexType>
bool CumMinMaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto element_size = (outer_size_ * inner_size_) * axis_size_;
  if (element_size == 0) {
    return true;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto input_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto value_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  auto index_ptr = GetDeviceAddress<IndexType>(outputs, kIndex1);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(cuda_stream, input_ptr, value_ptr, index_ptr)) {
    return false;
  }

  auto status = CumMinMax(cum_op_type_, input_ptr, value_ptr, index_ptr, outer_size_, axis_size_, inner_size_,
                          device_id_, cuda_stream);
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

// Note that in definition of primitive, Cummin return int32 as indices and Cummax return int64 as indices. (see
// cummax.cc and cummin.cc).
std::map<CumOpType, std::vector<std::pair<KernelAttr, CumMinMaxGpuKernelMod::CumMinMaxLaunchFunc>>>
  CumMinMaxGpuKernelMod::func_list_ = {
    {
      CUMMIN,
      {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<int8_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<int16_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<int32_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<int64_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<half, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<float, int32_t>},
       {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
        &CumMinMaxGpuKernelMod::LaunchKernel<double, int32_t>}},
    },
    {CUMMAX,
     {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<int8_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<int16_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<int32_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<int64_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<half, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<float, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxGpuKernelMod::LaunchKernel<double, int64_t>}}},
};

std::vector<KernelAttr> CumMinMaxGpuKernelMod::GetOpSupport() {
  auto iter = func_list_.find(cum_op_type_);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "CumMin/CumMax Something unexpected happened!";
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumMinMaxGpuKernelMod::CumMinMaxLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Cummin,
                                 []() { return std::make_shared<CumMinMaxGpuKernelMod>(CUMMIN); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Cummax,
                                 []() { return std::make_shared<CumMinMaxGpuKernelMod>(CUMMAX); });
}  // namespace kernel
}  // namespace mindspore
